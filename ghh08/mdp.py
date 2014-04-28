import random

default = (0.0, 0.0)

train = {0 : (-0.982730,  0.631781),
         1 : (-0.045686,  0.579628),
         2 : (-0.102530, -0.270641),
         3 : ( 0.762356,  0.283768),
         4 : (-0.489788,  0.569548),
         5 : ( 0.999573,  0.149759),
         6 : ( 0.887261,  0.382718),
         7 : (-0.150035, -0.841268),
         8 : ( 0.812929,  0.329428),
         9 : (-0.431899, -0.645667)}

test  = {0  : ( 0.602084,  0.697216),
         1  : (-0.620649,  0.528575),
         2  : (-0.736982, -0.449372),
         3  : (-0.075114,  0.312063),
         4  : (-0.091526,  0.276980),
         5  : (-0.486639,  0.123578),
         6  : (-0.260893,  0.084034),
         7  : ( 0.360428,  0.471989),
         8  : (-0.052433, -0.342526),
         9  : (-0.537700,  0.733001),
         10 : ( 0.746260, -0.167417),
         11 : (-0.695378, -0.579348),
         12 : (-0.518763, -0.531468),
         13 : (-0.078475, -0.751539),
         14 : (-0.280772, -0.855634),
         15 : ( 0.498471,  0.598713),
         16 : ( 0.940042, -0.847786),
         17 : ( 0.679303,  0.384155),
         18 : (-0.055490, -0.099921),
         19 : ( 0.790156, -0.855291),
         20 : ( 0.162346,  0.559800),
         21 : (-0.436883, -0.572197),
         22 : ( 0.929061,  0.884426),
         23 : (-0.774169,  0.575602),
         24 : (-0.592990, -0.719097),
         25 : (-0.536978, -0.733914),
         26 : ( 0.305824,  0.600994),
         27 : ( 0.767343,  0.682454),
         28 : (-0.770945,  0.586594),
         29 : (-0.375615,  0.192779),
         30 : ( 0.800311, -0.274053),
         31 : ( 0.502846, -0.348230),
         32 : ( 0.077578, -0.964760),
         33 : (-0.300474,  0.933242),
         34 : ( 0.842566,  0.573594),
         35 : (-0.706835, -0.432618),
         36 : (-0.687246,  0.898454),
         37 : ( 0.000431, -0.466892),
         38 : (-0.714204,  0.414413),
         39 : ( 0.457820,  0.479161),
         40 : (-0.351509, -0.172799),
         41 : ( 0.161959,  0.060760),
         42 : ( 0.360581, -0.849299),
         43 : (-0.243714,  0.138319),
         44 : ( 0.650950, -0.881746),
         45 : ( 0.096449,  0.770099),
         46 : ( 0.198432, -0.714164),
         47 : ( 0.238221, -0.285205),
         48 : (-0.248307,  0.007722),
         49 : ( 0.971504, -0.758884),
         50 : ( 0.739280, -0.376526),
         51 : (-0.266100, -0.129304),
         52 : (-0.347004,  0.326676),
         53 : (-0.041371,  0.089830),
         54 : ( 0.720418,  0.999560),
         55 : (-0.805017,  0.138119),
         56 : (-0.630552,  0.396823),
         57 : ( 0.111480, -0.709140),
         58 : ( 0.957411, -0.016304),
         59 : (-0.619700, -0.684330),
         60 : (-0.066451,  0.696918),
         61 : (-0.208537,  0.883721),
         62 : ( 0.852257,  0.621657),
         63 : (-0.059442,  0.509190),
         64 : (-0.579050,  0.137871),
         65 : (-0.646846,  0.802977),
         66 : ( 0.111974,  0.139250),
         67 : ( 0.539741,  0.772420),
         68 : (-0.690825, -0.518722),
         69 : (-0.788696,  0.256571),
         70 : ( 0.159520, -0.094063),
         71 : ( 0.339391,  0.536766),
         72 : ( 0.553304, -0.817504),
         73 : (-0.487198,  0.343453),
         74 : (-0.659916,  0.472813),
         75 : (-0.660172, -0.593397),
         76 : ( 0.004920, -0.571473),
         77 : ( 0.739787,  0.112989),
         78 : (-0.250535,  0.425561),
         79 : (-0.793942, -0.631749),
         80 : ( 0.051023,  0.572587),
         81 : ( 0.035730,  0.754754),
         82 : (-0.591959,  0.525348),
         83 : (-0.139520, -0.444840),
         84 : ( 0.464451, -0.407500),
         85 : ( 0.818338,  0.276639),
         86 : (-0.631646, -0.004912),
         87 : (-0.658360,  0.025400),
         88 : (-0.102925,  0.478231),
         89 : ( 0.184435,  0.975881),
         90 : ( 0.078850,  0.261982),
         91 : ( 0.289023, -0.838460),
         92 : ( 0.174900,  0.921453),
         93 : (-0.738064,  0.244688),
         94 : ( 0.905134,  0.121942),
         95 : (-0.026824,  0.350263),
         96 : (-0.066333,  0.605705),
         97 : (-0.062166, -0.597598),
         98 : ( 0.045236, -0.597935),
         99 : (-0.069014,  0.075972)}

def generate():
  return random.choice([-1, 1]) * random.random()

def sample():
  return random.choice(test)