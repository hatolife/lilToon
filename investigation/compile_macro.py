"""Compile the actual continued macro in a small HLSL caller. Requires DXC."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys

source = "Assets/lilToon/Shader/Includes/lil_common_frag.hlsl"
rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
text = subprocess.check_output(["git", "show", f"{rev}:{source}"], text=True)
start = text.index("    #define OVERRIDE_ANISOTROPY ")
macro = text[start:text.index("\n#endif", start)]
stub = """
struct Data {
    float3x3 TBN;
    float3 N,T,B,V,reflectionN,matcapN,matcap2ndN;
    float anisotropy,perceptualRoughness;
};
float _UseAnisotropy,_AnisotropyScale,_Anisotropy2Reflection,_Anisotropy2MatCap,_Anisotropy2MatCap2nd;
#define LIL_SAMPLE_AnisotropyTangentMap
#define LIL_SAMPLE_AnisotropyScaleMask
#define lilUnpackNormalScale(map,scale) (map.xyz)
#define lilGetAnisotropyNormalWS(n,t,b,v,a) (t+b)
"""
with TemporaryDirectory() as tmp:
    shader = Path(tmp)/"macro.hlsl"
    shader.write_text(stub + macro + """
float4 main(float3 n : NORMAL, float3 t : TEXCOORD0) : SV_Target {
    Data fd = (Data)0;
    fd.N = n;
    fd.TBN = float3x3(t,cross(n,t),n);
    OVERRIDE_ANISOTROPY
    return float4(fd.T+fd.B,1);
}
""")
    subprocess.run(["dxc","-T","ps_6_0","-Fo",str(shader.with_suffix(".dxil")),shader],check=True)
