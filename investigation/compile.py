"""DXC on PATH. Compile only the production tangent block as a pixel shader."""
import subprocess as sp
from pathlib import Path
from tempfile import TemporaryDirectory
import re
path="Assets/lilToon/Shader/Includes/lil_common_frag.hlsl"
for rev in ("e698a2f","3a5916ec80571f73afa6c1e506fe0c65e12afcda"):
    s=sp.check_output(["git","show",rev+":"+path],text=True)
    block=s.split("lilUnpackNormalScale(anisoTangentMap, 1.0);",1)[1].split("fd.anisotropy =",1)[0]
    block=block.replace(chr(92),"").replace("fd.","")
    func=sp.check_output(["git","show",rev+":"+path.replace("frag","functions")],text=True)
    helper="float3 lilOrthoNormalize"+func.split("float3 lilOrthoNormalize",1)[1].split("}",1)[0]+"}"
    with TemporaryDirectory() as tmp:
        src=Path(tmp)/"frame.hlsl"
        src.write_text(helper+"""
float4 main(float3 anisoTangent:A,float3 N:NORMAL,float3 t:TEXCOORD0,
            float3 b:TEXCOORD1,float3 n:TEXCOORD2):SV_Target {
float3x3 TBN=float3x3(t,b,n); float3 T,B;
"""+block+"return float4(T+B,1);}")
        for opt in ("-O3","-Od"):
            asm=src.with_suffix(".ll")
            sp.run(["dxc","-T","ps_6_0",opt,"-Fc",asm,"-Fo",str(src)+".dxil",src],check=True)
            body=asm.read_text().split("define void @main()",1)[1].split("}",1)[0]
            print(rev,opt,"SSA assignments",len(re.findall(r"^  %\d+ =",body,re.M)),
                  "conditional branches",body.count("br i1"),"rsqrt",body.count("Rsqrt"))
