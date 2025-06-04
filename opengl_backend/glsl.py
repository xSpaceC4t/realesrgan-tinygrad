from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite, extra_pm
from tinygrad.helpers import strip_parens
import math

def sign_extend(val:UOp, sext_am:int):
  return (UOp.where((val >> (sext_am - 1)) > 0, UOp.const(dtypes.uint32, 0xffffffff) << sext_am, UOp.const(dtypes.uint32, 0)) \
        | val.bitcast(dtypes.uint32)).bitcast(dtypes.int)

# store for char: buf[idx/4] <- (var << (idx%4)*8))
def packed_store(bidx:UOp, var:UOp):
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//var.dtype.itemsize))*UOp.const(dtypes.uint32, 8*var.dtype.itemsize)
  new_v = (var & (0xFF if var.dtype.itemsize == 1 else 0xFFFF)).cast(dtypes.uint32) << shift_am
  mask = (((0xFF if var.dtype.itemsize == 1 else 0xFFFF) << shift_am) ^ 0xFFFFFFFF).cast(dtypes.uint32)
  buf = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), dtype=dtypes.uint32)
  return UOp.store(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], bidx.src[1]//(4//var.dtype.itemsize))), ((buf & mask) | new_v.cast(dtypes.uint32)))

# load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
def packed_load(root:UOp, bidx:UOp, dtype:DType, var:UOp|None=None):
  div_idx = bidx.src[1]//(4//dtype.itemsize)
  shift_am = (bidx.src[1].cast(dtypes.uint32)%UOp.const(dtypes.uint32, 4//dtype.itemsize))*UOp.const(dtypes.uint32, 8*dtype.itemsize)
  if var is not None: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx, bidx.src[2])), var, dtype=dtypes.uint32, arg=root.arg)
  else: load = UOp.load(UOp(Ops.INDEX, bidx.dtype, (bidx.src[0], div_idx)), *root.src[1:], dtype=dtypes.uint32, arg=root.arg)
  val = (load.cast(dtypes.uint32) >> shift_am) & (0xFF if dtype.itemsize == 1 else 0xFFFF)
  return sign_extend(val, 8*dtype.itemsize).cast(dtype) if dtype in [dtypes.char, dtypes.short] else val.cast(dtype)

def is_packed(dt:DType) -> bool: return dt.itemsize < 4 and dt.base != dtypes.half

class OpenGLRenderer(CStyleLanguage):
  device = "OPENGL"

  # language options
  # kernel_prefix = "__kernel "
  # buffer_prefix = "__global "
  # smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "shared "
  # barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  barrier = "memoryBarrierShared();barrier();"
  float4 = "vec4"

  supports_float4 = False # add soon
  has_shared = True
  code_for_workitem = {
    "g": lambda x: f"int(gl_WorkGroupID.{'xyz'[int(x)]})", 
    "l": lambda x: f"int(gl_LocalInvocationID.{'xyz'[int(x)]})", 
    "i": lambda x: f"int(gl_GlobalInvocationID.{'xyz'[int(x)]})"
  }
  # type_map = { dtypes.int8: "char", dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong",
  #             dtypes.bfloat16: "ushort" }
  type_map = { dtypes.int8: "char", dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong",
              dtypes.bfloat16: "ushort", dtypes.int64: "int64_t" }
  code_for_op: dict = {
    Ops.SQRT: lambda x,dtype: f"sqrt({x})", Ops.RECIP: lambda x,dtype: f"(1/{x})", Ops.NEG: lambda x,dtype: f"-{x}",
    Ops.EXP2: lambda x,dtype: f"exp2({x})", Ops.LOG2: lambda x,dtype: f"log2({x})", Ops.SIN: lambda x,dtype: f"sin({x})",
    Ops.AND: lambda a,b,dtype: f"(int({a})&int({b}))", Ops.XOR: lambda a,b,dtype: f"({a}^{b})", Ops.OR: lambda a,b,dtype: f"({a}|{b})",
    Ops.ADD: lambda a,b,dtype: f"({a}+{b})", Ops.SUB: lambda a,b,dtype: f"({a}-{b})", Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
    Ops.MOD: lambda a,b,dtype: f"({a}%{b})", Ops.IDIV: lambda a,b,dtype: f"({a}/{b})", Ops.CMPNE: lambda a,b,dtype: f"(int({a})!=int({b}))",
    Ops.SHR: lambda a,b,dtype: f"({a}>>{b})", Ops.SHL: lambda a,b,dtype: f"({a}<<{b})", Ops.CMPLT: lambda a,b,dtype: f"({a}<{b})",
    Ops.WHERE: lambda a,b,c,dtype: f"(bool({a})?{b}:{c})" }

  string_rewrite = PatternMatcher([
    (UPat(Ops.WHERE, src=(UPat.var("a"), UPat.var("b"), UPat.var("c")), name="d"),
     lambda ctx,a,b,c,d: f"{ctx.render_dtype(d.dtype)}(bool({ctx[a]})?{ctx[b]}:{ctx[c]})"),
    (UPat(Ops.AND, src=(UPat.var("a"), UPat.var("b")), name="c"),
     lambda ctx,a,b,c: f"{ctx.render_dtype(c.dtype)}(int({ctx[a]})&int({ctx[b]}))"),


    # (UPat(Ops.VECTORIZE, name="x"),
    #  lambda ctx,x: f"{ctx.float4.replace('float4', ctx.render_dtype(x.dtype))}" + \
    #  f"{ctx.float4_style[0]}{','.join([ctx[y] for y in x.src])}{ctx.float4_style[1]}"),

    (UPat(Ops.CONST, dtype=dtypes.int64, name="x"), lambda ctx,x: f"{x.arg}"),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(), UPat(), UPat.var("gate"))).or_casted("bidx"), UPat.var("var")), allow_any_len=True),
     lambda ctx,bidx,var,gate: f"(bool({ctx[gate]})?{ctx[bidx]}:{ctx[var]})"),
    (UPat(Ops.BITCAST, dtype=dtypes.float, name="x", src=(UPat(dtype=dtypes.uint))), lambda ctx,x: f"uintBitsToFloat({ctx[x.src[0]]})"),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True), lambda ctx,bidx: f"{ctx[bidx]}"),
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var('idx')), allow_any_len=True),
     lambda ctx,buf,idx: f"{ctx[buf]}[{strip_parens(ctx[idx]) if idx.arg == Ops.ADD else ctx[idx]}]"),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True), lambda ctx,bidx,var: f"{ctx[bidx]} = {ctx[var]};"),
  ]) + base_rewrite

  def render_dtype(self, dt:DType, mutable=True) -> str:
      if isinstance(dt, PtrDType):
        return self.render_dtype(dt.base)
      if dt.count > 1: return self.type_map.get(scalar:=dt.scalar(), scalar.name).replace(" ", "_") + str(dt.count)
      return self.type_map.get(scalar:=dt.scalar(), scalar.name)

  def render_cast(self, dt:DType, val: str) -> str: return f"{self.render_dtype(dt)}({val})"

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    local_size = [num for _, num in sorted([u.arg for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == 'l'], key=lambda x: x[0])]
    if not local_size: local_size = [1]
    for i in range(3 - len(local_size)):
      local_size.append(1)

    prg = "#version 430\n#extension GL_ARB_gpu_shader_int64 : enable\n"
    prg += f"layout(local_size_x={local_size[0]}, local_size_y={local_size[1]}, local_size_z={local_size[2]}) in;\n"

    i = 0
    for name,(dtype,mutable) in bufs:
      ssbo = f"layout(std430, binding={i}) buffer Buffer{i} {{"
      ssbo += f" {self.render_dtype(dtype, mutable)} {name}[]; "
      ssbo += "};\n"
      prg += ssbo
      i += 1

    for line in kernel:
      if "shared" in line:
        prg += line

    prg += ''.join([f"{self.kernel_prefix}void main(",] +
    [") {\n"] + ['\n'.join(line for line in kernel if 'shared' not in line), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"