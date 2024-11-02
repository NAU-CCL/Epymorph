"""
This is data drawn from Pei's data files.
We're including it as a temporary measure, since we don't have
an ADRIO for humidity yet, and because the ADRIOs we do have
produce data values that differ from these.
It would also be feasible to use CSV ADRIOs to load this data
directly. But since a lot of epymorph uses Pei as an example
this is a convenience, and expected to be temporary.
"""

import numpy as np

from epymorph.data_type import CentroidDType
from epymorph.geography.us_census import StateScope

pei_scope = StateScope.in_states(states=["FL", "GA", "MD", "NC", "SC", "VA"], year=2015)

pei_centroids = np.array(
    [
        (-81.5158, 27.6648),
        (-82.9071, 32.1574),
        (-76.6413, 39.0458),
        (-79.0193, 35.7596),
        (-81.1637, 33.8361),
        (-78.6569, 37.4316),
    ],
    dtype=CentroidDType,
)

pei_population = np.array(
    [18811310, 9687653, 5773552, 9535483, 4625364, 8001024], dtype=np.int64
)

pei_commuters = np.array(
    [
        [7993452, 13805, 2410, 2938, 1783, 3879],
        [15066, 4091461, 966, 6057, 20318, 2147],
        [949, 516, 2390255, 947, 91, 122688],
        [3005, 5730, 1872, 4121984, 38081, 29487],
        [1709, 23513, 630, 64872, 1890853, 1620],
        [1368, 1175, 68542, 16869, 577, 3567788],
    ],
    dtype=np.int64,
)

# humidity in 2015 (365 days)
pei_humidity = np.array(
    [
        [0.01003, 0.008144, 0.004738, 0.00681, 0.007467, 0.004909],
        [0.0105, 0.008211, 0.004495, 0.006562, 0.007302, 0.004716],
        [0.010631, 0.008169, 0.0053, 0.007162, 0.007736, 0.005407],
        [0.010797, 0.007956, 0.0054, 0.00712, 0.00776, 0.005611],
        [0.010727, 0.008105, 0.005233, 0.00726, 0.007819, 0.005504],
        [0.01009, 0.008425, 0.004814, 0.007089, 0.007881, 0.005119],
        [0.009815, 0.008386, 0.005, 0.007195, 0.007867, 0.00521],
        [0.010363, 0.007858, 0.004857, 0.006825, 0.007343, 0.00491],
        [0.010242, 0.007508, 0.0047, 0.006633, 0.007217, 0.00473],
        [0.009609, 0.007048, 0.004052, 0.006199, 0.006764, 0.004276],
        [0.009782, 0.00739, 0.00429, 0.006251, 0.006876, 0.004457],
        [0.009573, 0.007775, 0.004914, 0.00681, 0.007402, 0.005012],
        [0.009151, 0.008051, 0.005257, 0.00729, 0.007793, 0.005478],
        [0.009114, 0.007194, 0.004824, 0.006514, 0.007036, 0.004898],
        [0.009586, 0.00701, 0.004419, 0.006087, 0.006702, 0.004422],
        [0.009771, 0.006976, 0.004324, 0.006073, 0.006602, 0.004267],
        [0.009916, 0.007182, 0.003924, 0.005774, 0.006512, 0.00395],
        [0.00951, 0.007415, 0.004476, 0.006351, 0.006874, 0.004434],
        [0.009165, 0.007085, 0.003914, 0.005829, 0.006433, 0.004002],
        [0.009451, 0.006524, 0.003748, 0.005594, 0.006214, 0.003876],
        [0.00939, 0.007217, 0.003633, 0.005592, 0.006452, 0.003741],
        [0.009205, 0.00759, 0.00381, 0.005946, 0.006779, 0.004008],
        [0.008819, 0.007765, 0.004548, 0.006529, 0.007188, 0.00471],
        [0.009298, 0.007635, 0.004833, 0.006829, 0.007386, 0.004962],
        [0.010034, 0.006387, 0.004024, 0.005414, 0.005981, 0.003903],
        [0.009865, 0.006424, 0.003424, 0.004979, 0.005731, 0.003404],
        [0.009244, 0.006912, 0.003743, 0.005405, 0.006252, 0.00379],
        [0.009452, 0.007364, 0.004395, 0.006227, 0.006857, 0.004595],
        [0.009922, 0.008088, 0.003971, 0.006337, 0.007169, 0.00427],
        [0.010102, 0.008568, 0.004552, 0.006879, 0.00766, 0.004733],
        [0.00967, 0.00694, 0.003957, 0.005821, 0.006364, 0.00393],
        [0.009465, 0.00713, 0.004038, 0.005765, 0.006412, 0.004019],
        [0.009473, 0.007817, 0.00411, 0.006108, 0.0069, 0.004237],
        [0.009869, 0.007693, 0.004205, 0.006364, 0.006993, 0.004406],
        [0.009859, 0.007357, 0.004557, 0.006596, 0.007105, 0.0048],
        [0.009472, 0.006649, 0.003981, 0.005577, 0.00616, 0.004025],
        [0.00972, 0.007214, 0.003967, 0.005705, 0.006469, 0.004114],
        [0.009781, 0.007535, 0.004129, 0.006332, 0.006967, 0.0044],
        [0.008998, 0.006993, 0.003995, 0.005981, 0.006588, 0.004349],
        [0.008965, 0.007864, 0.0043, 0.00638, 0.007248, 0.004713],
        [0.009688, 0.008125, 0.0045, 0.006562, 0.00735, 0.004655],
        [0.009386, 0.008112, 0.004081, 0.006412, 0.007248, 0.004393],
        [0.009492, 0.007551, 0.003886, 0.006048, 0.006869, 0.004161],
        [0.009803, 0.007173, 0.003852, 0.00592, 0.006648, 0.004171],
        [0.009468, 0.007719, 0.004324, 0.006504, 0.007162, 0.004708],
        [0.00919, 0.00809, 0.004238, 0.0067, 0.007405, 0.00477],
        [0.009572, 0.008445, 0.004719, 0.007239, 0.007883, 0.005179],
        [0.010095, 0.008001, 0.004143, 0.006375, 0.007126, 0.004413],
        [0.01012, 0.00777, 0.0044, 0.00623, 0.006993, 0.004546],
        [0.010347, 0.007843, 0.00461, 0.0065, 0.007074, 0.004821],
        [0.010309, 0.008111, 0.004719, 0.006764, 0.007474, 0.005031],
        [0.010009, 0.008877, 0.00489, 0.007371, 0.008162, 0.005248],
        [0.010024, 0.009679, 0.005, 0.007724, 0.008545, 0.00542],
        [0.010101, 0.008848, 0.004862, 0.007165, 0.007981, 0.005105],
        [0.010214, 0.00791, 0.004452, 0.006431, 0.007071, 0.004664],
        [0.010278, 0.008018, 0.004086, 0.006123, 0.00694, 0.004479],
        [0.010087, 0.008654, 0.004448, 0.006748, 0.007605, 0.004894],
        [0.010355, 0.008181, 0.004571, 0.006644, 0.007498, 0.005017],
        [0.010427, 0.008333, 0.004762, 0.007206, 0.007821, 0.005261],
        [0.011015, 0.008748, 0.004338, 0.00699, 0.007924, 0.00494],
        [0.010848, 0.009258, 0.005414, 0.007926, 0.008529, 0.00593],
        [0.010208, 0.008746, 0.00509, 0.007583, 0.008217, 0.005597],
        [0.009757, 0.008039, 0.004595, 0.006887, 0.007631, 0.005057],
        [0.010093, 0.00801, 0.004838, 0.006923, 0.007574, 0.005283],
        [0.010081, 0.008942, 0.005186, 0.007589, 0.008407, 0.005759],
        [0.010327, 0.009295, 0.005267, 0.007867, 0.008633, 0.005875],
        [0.010199, 0.008657, 0.005767, 0.007563, 0.008121, 0.005987],
        [0.010143, 0.008652, 0.00511, 0.007189, 0.007998, 0.00551],
        [0.009692, 0.008573, 0.005038, 0.007301, 0.008052, 0.005444],
        [0.009769, 0.00821, 0.00491, 0.007044, 0.007712, 0.00533],
        [0.009577, 0.008525, 0.004795, 0.007027, 0.00786, 0.005278],
        [0.009547, 0.009267, 0.005524, 0.007917, 0.008676, 0.006165],
        [0.009643, 0.009277, 0.005695, 0.008032, 0.008738, 0.006319],
        [0.009895, 0.00928, 0.005729, 0.007868, 0.008574, 0.006176],
        [0.010718, 0.009733, 0.00549, 0.008096, 0.008912, 0.00613],
        [0.010884, 0.009432, 0.005295, 0.007907, 0.00864, 0.005974],
        [0.011186, 0.009346, 0.005052, 0.007533, 0.008398, 0.005693],
        [0.011291, 0.009375, 0.005414, 0.007588, 0.008374, 0.005939],
        [0.011956, 0.009446, 0.005329, 0.007596, 0.008531, 0.005845],
        [0.011348, 0.009345, 0.0056, 0.007965, 0.008719, 0.006239],
        [0.010864, 0.008571, 0.005567, 0.007408, 0.008069, 0.005916],
        [0.010934, 0.008793, 0.005129, 0.007307, 0.008095, 0.005749],
        [0.011081, 0.009018, 0.005414, 0.007615, 0.0084, 0.006056],
        [0.010876, 0.009925, 0.005381, 0.008106, 0.009105, 0.006285],
        [0.010844, 0.010088, 0.006033, 0.008535, 0.009357, 0.006739],
        [0.011068, 0.010262, 0.006343, 0.008819, 0.009569, 0.006995],
        [0.011229, 0.010521, 0.006876, 0.009365, 0.01, 0.007534],
        [0.011388, 0.010819, 0.006719, 0.009286, 0.010055, 0.007323],
        [0.0113, 0.010336, 0.006514, 0.008746, 0.009414, 0.006969],
        [0.011776, 0.010332, 0.006338, 0.008826, 0.00966, 0.007092],
        [0.011904, 0.01025, 0.007038, 0.009333, 0.009902, 0.007741],
        [0.01186, 0.009836, 0.006876, 0.008806, 0.009355, 0.007369],
        [0.011244, 0.009998, 0.006767, 0.008824, 0.009552, 0.007385],
        [0.010541, 0.010074, 0.006686, 0.008862, 0.009548, 0.007277],
        [0.010939, 0.009864, 0.005638, 0.008251, 0.009198, 0.006492],
        [0.011814, 0.010223, 0.006452, 0.008811, 0.009574, 0.00719],
        [0.011852, 0.010324, 0.007048, 0.009201, 0.009843, 0.007644],
        [0.01168, 0.010515, 0.00709, 0.00932, 0.010024, 0.007863],
        [0.011676, 0.010502, 0.007014, 0.009199, 0.009802, 0.007687],
        [0.012229, 0.01023, 0.006771, 0.009155, 0.009795, 0.007703],
        [0.012468, 0.010901, 0.006771, 0.009512, 0.010455, 0.00785],
        [0.011615, 0.011148, 0.007324, 0.009926, 0.010633, 0.008174],
        [0.011394, 0.01088, 0.007038, 0.009379, 0.010164, 0.007643],
        [0.01188, 0.01056, 0.006919, 0.009238, 0.010031, 0.007629],
        [0.01212, 0.011138, 0.007829, 0.010204, 0.01085, 0.008755],
        [0.012775, 0.010276, 0.008143, 0.010008, 0.010395, 0.008739],
        [0.012888, 0.010226, 0.007705, 0.009751, 0.010245, 0.008417],
        [0.012084, 0.010138, 0.007081, 0.009196, 0.00986, 0.00785],
        [0.011739, 0.010814, 0.007829, 0.009602, 0.010402, 0.00853],
        [0.010981, 0.01149, 0.007981, 0.010546, 0.01119, 0.009008],
        [0.011076, 0.01181, 0.008405, 0.011112, 0.011698, 0.009591],
        [0.011556, 0.011177, 0.008705, 0.010769, 0.011102, 0.00921],
        [0.012226, 0.011181, 0.008095, 0.009817, 0.01056, 0.008492],
        [0.012489, 0.01112, 0.007462, 0.009956, 0.01076, 0.008383],
        [0.012573, 0.011604, 0.007981, 0.010581, 0.011143, 0.009022],
        [0.012186, 0.01162, 0.008738, 0.010721, 0.011219, 0.009469],
        [0.012329, 0.011424, 0.008381, 0.010532, 0.011083, 0.008934],
        [0.012918, 0.011377, 0.008019, 0.010204, 0.0109, 0.008645],
        [0.012835, 0.011214, 0.007619, 0.009918, 0.01065, 0.008534],
        [0.013082, 0.011663, 0.008267, 0.010499, 0.011286, 0.009168],
        [0.013213, 0.012345, 0.008943, 0.011529, 0.012069, 0.009829],
        [0.012699, 0.012381, 0.0092, 0.011644, 0.012226, 0.010143],
        [0.012721, 0.012861, 0.008586, 0.011723, 0.012457, 0.009722],
        [0.013005, 0.012939, 0.008467, 0.011207, 0.012029, 0.009374],
        [0.013631, 0.012986, 0.008948, 0.011664, 0.012288, 0.00998],
        [0.013694, 0.012804, 0.009138, 0.01196, 0.012352, 0.010051],
        [0.01355, 0.012889, 0.008576, 0.011577, 0.01224, 0.009653],
        [0.013564, 0.013223, 0.009176, 0.012142, 0.012743, 0.010309],
        [0.013882, 0.01323, 0.009562, 0.011998, 0.012795, 0.01065],
        [0.014093, 0.013804, 0.009619, 0.012176, 0.013069, 0.010587],
        [0.013752, 0.013831, 0.009219, 0.011769, 0.012907, 0.010207],
        [0.013953, 0.013202, 0.009905, 0.011927, 0.012564, 0.010598],
        [0.014349, 0.013444, 0.009295, 0.011729, 0.01264, 0.010247],
        [0.014904, 0.01332, 0.008795, 0.011281, 0.012262, 0.009755],
        [0.014976, 0.013571, 0.009576, 0.012142, 0.012924, 0.010584],
        [0.014882, 0.014007, 0.009943, 0.012442, 0.013255, 0.010955],
        [0.014569, 0.014274, 0.009952, 0.012763, 0.013595, 0.011038],
        [0.014215, 0.013798, 0.01009, 0.012668, 0.01326, 0.011097],
        [0.014428, 0.01332, 0.009767, 0.012045, 0.01279, 0.010514],
        [0.014867, 0.013473, 0.009214, 0.012323, 0.012969, 0.010358],
        [0.015107, 0.013651, 0.009238, 0.012152, 0.013064, 0.010161],
        [0.014847, 0.013617, 0.009352, 0.01178, 0.01279, 0.010035],
        [0.014578, 0.01397, 0.009648, 0.011879, 0.012952, 0.010241],
        [0.01471, 0.014381, 0.010433, 0.012475, 0.013538, 0.011157],
        [0.015124, 0.01476, 0.010729, 0.013529, 0.014329, 0.011998],
        [0.015488, 0.015054, 0.011138, 0.013985, 0.014736, 0.012324],
        [0.015705, 0.015443, 0.010671, 0.013993, 0.014869, 0.012063],
        [0.016168, 0.015455, 0.010195, 0.013577, 0.01459, 0.011706],
        [0.016258, 0.015467, 0.010143, 0.013595, 0.014571, 0.011665],
        [0.016137, 0.01536, 0.010371, 0.013549, 0.014576, 0.011645],
        [0.016281, 0.015662, 0.011629, 0.013974, 0.014869, 0.012527],
        [0.016463, 0.015933, 0.011581, 0.014226, 0.015114, 0.012542],
        [0.016545, 0.016235, 0.011571, 0.014575, 0.015352, 0.012759],
        [0.016747, 0.01701, 0.011548, 0.015126, 0.016069, 0.012768],
        [0.017063, 0.017375, 0.01111, 0.015449, 0.016481, 0.01281],
        [0.017179, 0.016867, 0.011395, 0.015158, 0.015974, 0.012788],
        [0.017423, 0.016612, 0.011557, 0.01472, 0.015579, 0.012645],
        [0.017638, 0.016349, 0.011629, 0.014689, 0.015538, 0.012802],
        [0.017717, 0.016383, 0.012357, 0.015243, 0.01585, 0.013627],
        [0.017514, 0.016749, 0.012662, 0.015337, 0.015993, 0.013729],
        [0.017574, 0.01716, 0.012371, 0.015506, 0.016457, 0.013745],
        [0.017798, 0.017605, 0.012138, 0.015517, 0.016602, 0.013619],
        [0.017733, 0.0174, 0.013133, 0.01583, 0.016581, 0.014205],
        [0.017653, 0.017294, 0.013429, 0.015871, 0.0165, 0.014418],
        [0.017737, 0.017733, 0.013586, 0.016124, 0.016917, 0.014603],
        [0.017914, 0.017887, 0.0136, 0.016571, 0.017136, 0.014755],
        [0.018387, 0.017757, 0.013424, 0.016127, 0.016855, 0.014476],
        [0.018591, 0.017573, 0.013243, 0.01588, 0.016652, 0.01407],
        [0.018337, 0.017419, 0.013562, 0.01572, 0.016474, 0.014347],
        [0.018416, 0.017435, 0.014281, 0.016476, 0.016862, 0.014972],
        [0.018418, 0.017732, 0.014071, 0.016525, 0.016983, 0.014682],
        [0.01849, 0.017845, 0.013833, 0.016314, 0.01694, 0.01445],
        [0.018627, 0.01799, 0.014005, 0.016806, 0.017269, 0.014918],
        [0.018777, 0.018045, 0.013281, 0.016939, 0.017367, 0.014484],
        [0.018698, 0.017938, 0.013533, 0.016652, 0.017305, 0.014382],
        [0.018778, 0.018239, 0.014281, 0.016906, 0.017567, 0.015076],
        [0.018799, 0.018431, 0.014938, 0.017411, 0.017867, 0.015658],
        [0.0188, 0.018575, 0.015324, 0.017595, 0.018036, 0.015844],
        [0.018929, 0.018767, 0.014938, 0.017563, 0.018055, 0.015785],
        [0.018939, 0.018812, 0.015005, 0.017618, 0.018124, 0.015857],
        [0.019059, 0.01862, 0.014852, 0.0176, 0.017905, 0.01557],
        [0.019257, 0.018618, 0.014738, 0.017451, 0.017833, 0.015652],
        [0.019416, 0.018546, 0.01409, 0.016893, 0.017538, 0.014951],
        [0.019318, 0.018471, 0.014362, 0.016687, 0.017414, 0.01519],
        [0.019271, 0.018512, 0.015429, 0.017065, 0.017688, 0.015943],
        [0.019285, 0.019001, 0.015671, 0.018065, 0.018498, 0.016567],
        [0.019295, 0.019383, 0.014952, 0.018432, 0.0188, 0.016085],
        [0.019361, 0.01918, 0.014633, 0.017771, 0.018276, 0.015465],
        [0.019265, 0.018945, 0.015171, 0.017794, 0.01835, 0.015837],
        [0.019379, 0.01916, 0.015352, 0.01847, 0.018836, 0.016407],
        [0.01959, 0.019505, 0.015352, 0.018842, 0.019202, 0.016754],
        [0.019758, 0.019793, 0.014095, 0.01862, 0.019119, 0.015783],
        [0.019676, 0.019568, 0.013995, 0.017971, 0.018595, 0.015486],
        [0.01963, 0.01934, 0.015057, 0.01792, 0.018467, 0.015984],
        [0.019581, 0.019292, 0.01559, 0.018035, 0.0184, 0.016158],
        [0.019635, 0.019188, 0.015752, 0.018145, 0.018355, 0.016318],
        [0.019547, 0.019318, 0.015776, 0.018268, 0.018507, 0.016453],
        [0.019677, 0.019464, 0.016305, 0.018319, 0.0186, 0.01659],
        [0.019778, 0.019404, 0.016843, 0.018613, 0.018843, 0.017079],
        [0.019819, 0.019561, 0.017024, 0.018929, 0.018955, 0.017431],
        [0.019768, 0.019676, 0.016262, 0.019014, 0.019079, 0.016991],
        [0.019957, 0.019755, 0.016214, 0.018649, 0.019036, 0.016726],
        [0.019959, 0.019654, 0.016371, 0.019062, 0.019221, 0.017201],
        [0.02008, 0.019857, 0.016781, 0.019207, 0.019376, 0.017295],
        [0.020069, 0.019738, 0.016281, 0.018985, 0.019081, 0.016856],
        [0.019953, 0.019532, 0.016119, 0.018708, 0.018881, 0.016875],
        [0.019765, 0.019636, 0.015905, 0.018779, 0.019062, 0.016878],
        [0.019587, 0.019821, 0.016633, 0.018944, 0.01919, 0.017198],
        [0.019693, 0.019948, 0.01649, 0.019085, 0.019395, 0.01717],
        [0.019741, 0.019862, 0.016748, 0.019115, 0.019271, 0.01723],
        [0.019812, 0.019871, 0.016481, 0.018808, 0.018983, 0.016848],
        [0.019936, 0.019561, 0.016533, 0.018396, 0.018729, 0.016825],
        [0.020027, 0.019321, 0.01621, 0.017937, 0.018288, 0.016388],
        [0.020056, 0.019204, 0.016114, 0.017727, 0.018195, 0.016195],
        [0.020058, 0.018943, 0.016357, 0.017985, 0.018236, 0.01649],
        [0.019923, 0.019195, 0.017005, 0.018223, 0.018424, 0.016849],
        [0.019809, 0.019407, 0.016114, 0.018281, 0.018562, 0.016419],
        [0.019729, 0.019442, 0.015133, 0.017976, 0.018464, 0.015677],
        [0.019728, 0.019005, 0.014519, 0.017419, 0.017838, 0.015188],
        [0.019813, 0.018613, 0.014938, 0.01724, 0.017655, 0.015381],
        [0.019859, 0.018724, 0.015081, 0.017532, 0.017969, 0.015801],
        [0.019646, 0.018875, 0.015852, 0.017913, 0.018271, 0.016219],
        [0.019451, 0.019054, 0.015967, 0.018046, 0.018355, 0.016357],
        [0.019521, 0.01917, 0.01589, 0.01786, 0.018286, 0.0162],
        [0.019614, 0.019102, 0.016219, 0.018123, 0.018417, 0.016581],
        [0.019803, 0.019245, 0.0161, 0.017764, 0.018279, 0.016318],
        [0.019889, 0.018998, 0.015933, 0.017923, 0.018195, 0.016347],
        [0.019893, 0.019064, 0.016343, 0.018067, 0.018314, 0.016693],
        [0.019912, 0.019431, 0.016376, 0.018367, 0.018717, 0.016588],
        [0.019899, 0.019743, 0.016029, 0.018602, 0.01899, 0.016446],
        [0.019772, 0.019589, 0.015324, 0.018263, 0.018719, 0.016158],
        [0.019801, 0.01932, 0.015238, 0.017868, 0.018388, 0.016135],
        [0.019721, 0.018933, 0.01469, 0.017599, 0.018117, 0.01544],
        [0.019832, 0.018704, 0.014095, 0.017154, 0.017752, 0.014774],
        [0.019828, 0.018601, 0.013924, 0.016846, 0.017583, 0.014615],
        [0.019707, 0.018561, 0.0143, 0.016846, 0.017605, 0.01487],
        [0.019564, 0.018521, 0.014405, 0.016855, 0.01746, 0.015111],
        [0.019482, 0.018535, 0.015048, 0.017351, 0.017621, 0.015788],
        [0.019441, 0.018825, 0.015595, 0.018055, 0.018105, 0.016333],
        [0.01953, 0.019002, 0.015924, 0.018292, 0.018412, 0.016667],
        [0.019526, 0.018902, 0.015895, 0.018045, 0.018238, 0.016474],
        [0.019456, 0.018877, 0.01501, 0.01774, 0.017938, 0.01577],
        [0.019503, 0.018587, 0.014438, 0.017474, 0.017679, 0.015358],
        [0.019586, 0.018382, 0.013662, 0.017221, 0.017471, 0.014636],
        [0.019592, 0.017979, 0.013338, 0.01603, 0.01671, 0.013811],
        [0.019547, 0.017705, 0.013729, 0.016171, 0.016631, 0.01401],
        [0.019416, 0.017282, 0.013795, 0.015801, 0.01621, 0.013838],
        [0.019318, 0.016994, 0.012705, 0.015425, 0.015893, 0.013135],
        [0.019224, 0.017236, 0.0128, 0.015686, 0.016176, 0.013503],
        [0.018935, 0.017605, 0.013386, 0.01611, 0.016626, 0.013883],
        [0.019005, 0.017586, 0.013467, 0.016296, 0.016707, 0.013973],
        [0.018855, 0.017298, 0.0137, 0.015658, 0.016107, 0.013819],
        [0.018807, 0.017052, 0.01341, 0.01561, 0.015907, 0.013544],
        [0.018779, 0.016719, 0.01211, 0.014786, 0.015345, 0.012363],
        [0.018691, 0.016757, 0.01109, 0.014325, 0.015243, 0.011609],
        [0.018516, 0.016698, 0.012248, 0.014795, 0.015457, 0.012483],
        [0.018457, 0.01703, 0.013519, 0.01575, 0.016131, 0.013792],
        [0.018529, 0.016998, 0.013286, 0.015783, 0.016052, 0.01343],
        [0.018568, 0.016583, 0.013062, 0.015215, 0.015533, 0.012994],
        [0.018709, 0.016536, 0.012895, 0.014874, 0.015274, 0.012854],
        [0.018651, 0.016313, 0.012319, 0.014576, 0.01499, 0.01241],
        [0.018505, 0.016187, 0.011838, 0.014393, 0.014907, 0.012262],
        [0.018201, 0.016365, 0.01129, 0.014238, 0.015017, 0.011736],
        [0.018067, 0.016717, 0.011057, 0.014357, 0.015336, 0.011732],
        [0.018144, 0.016593, 0.012329, 0.014901, 0.015471, 0.012486],
        [0.01837, 0.016233, 0.012033, 0.014363, 0.014819, 0.011982],
        [0.017969, 0.015974, 0.011019, 0.013906, 0.014643, 0.011401],
        [0.01779, 0.015877, 0.011129, 0.014383, 0.014836, 0.011667],
        [0.017952, 0.015788, 0.012033, 0.014351, 0.014821, 0.012109],
        [0.018156, 0.015985, 0.012652, 0.014582, 0.014962, 0.012543],
        [0.018203, 0.015279, 0.011995, 0.014088, 0.014443, 0.011795],
        [0.017879, 0.014385, 0.010195, 0.01255, 0.013124, 0.010127],
        [0.017754, 0.013657, 0.009519, 0.011694, 0.012298, 0.009417],
        [0.017408, 0.013742, 0.009186, 0.011564, 0.012293, 0.009094],
        [0.017207, 0.013761, 0.009133, 0.011281, 0.012224, 0.009179],
        [0.017103, 0.014143, 0.009952, 0.011963, 0.01265, 0.009793],
        [0.017158, 0.01447, 0.010776, 0.012549, 0.013079, 0.010691],
        [0.016928, 0.014414, 0.010376, 0.01274, 0.013314, 0.010512],
        [0.016424, 0.014601, 0.009376, 0.012362, 0.013088, 0.009622],
        [0.016401, 0.013867, 0.009086, 0.01172, 0.012445, 0.009208],
        [0.016422, 0.013846, 0.009429, 0.012001, 0.012705, 0.009488],
        [0.016831, 0.01424, 0.010424, 0.01249, 0.0131, 0.010379],
        [0.017054, 0.014279, 0.0105, 0.012885, 0.013162, 0.01027],
        [0.016439, 0.013832, 0.009552, 0.011918, 0.012505, 0.009402],
        [0.015825, 0.013394, 0.009162, 0.011118, 0.011826, 0.00885],
        [0.015627, 0.013168, 0.008862, 0.011094, 0.011795, 0.008933],
        [0.015871, 0.012768, 0.009152, 0.01126, 0.011788, 0.009214],
        [0.016074, 0.011694, 0.008719, 0.010502, 0.010838, 0.00835],
        [0.015771, 0.011519, 0.008514, 0.010082, 0.010536, 0.008058],
        [0.015614, 0.01148, 0.008224, 0.010127, 0.010521, 0.008031],
        [0.015051, 0.01197, 0.008438, 0.010462, 0.01101, 0.008318],
        [0.014631, 0.011526, 0.008386, 0.010214, 0.010748, 0.008249],
        [0.014186, 0.011938, 0.008162, 0.010451, 0.011, 0.00822],
        [0.014418, 0.011642, 0.00779, 0.0098, 0.0105, 0.007809],
        [0.014457, 0.011927, 0.007448, 0.009762, 0.01065, 0.00757],
        [0.01448, 0.011419, 0.007814, 0.009425, 0.010248, 0.007595],
        [0.014345, 0.011358, 0.008005, 0.009983, 0.010612, 0.008008],
        [0.014243, 0.011212, 0.00809, 0.010363, 0.010686, 0.008071],
        [0.014039, 0.011039, 0.007929, 0.010029, 0.010319, 0.00797],
        [0.014096, 0.011281, 0.008614, 0.010179, 0.010555, 0.008367],
        [0.014065, 0.01112, 0.007543, 0.009998, 0.010548, 0.00767],
        [0.013486, 0.010419, 0.005981, 0.008287, 0.009179, 0.00608],
        [0.013031, 0.010869, 0.006552, 0.008895, 0.009645, 0.006668],
        [0.013697, 0.011132, 0.007548, 0.009493, 0.010069, 0.007393],
        [0.013997, 0.01123, 0.007862, 0.009807, 0.010319, 0.007778],
        [0.013559, 0.010918, 0.007638, 0.009696, 0.010207, 0.007561],
        [0.013225, 0.010029, 0.007048, 0.008812, 0.009221, 0.006808],
        [0.013519, 0.010074, 0.006552, 0.008718, 0.009255, 0.00652],
        [0.013245, 0.01003, 0.006771, 0.008704, 0.009281, 0.006673],
        [0.013193, 0.009656, 0.006933, 0.008676, 0.009012, 0.006658],
        [0.013148, 0.00938, 0.006262, 0.008138, 0.008717, 0.006098],
        [0.012884, 0.009192, 0.006424, 0.007794, 0.008424, 0.006136],
        [0.012749, 0.009283, 0.006176, 0.007888, 0.008529, 0.006156],
        [0.011884, 0.009677, 0.006767, 0.0087, 0.009155, 0.006664],
        [0.01148, 0.009482, 0.006638, 0.008281, 0.008798, 0.006473],
        [0.011775, 0.009177, 0.006267, 0.007915, 0.008457, 0.006019],
        [0.011703, 0.009302, 0.006295, 0.007898, 0.00845, 0.005938],
        [0.01181, 0.009289, 0.006948, 0.008345, 0.008729, 0.006593],
        [0.012207, 0.009271, 0.006995, 0.00816, 0.008607, 0.006561],
        [0.012137, 0.009202, 0.006443, 0.008363, 0.008776, 0.00652],
        [0.011707, 0.008574, 0.005976, 0.007802, 0.008229, 0.005899],
        [0.011552, 0.0084, 0.005457, 0.0072, 0.007814, 0.00548],
        [0.012004, 0.008992, 0.00609, 0.007875, 0.008471, 0.006175],
        [0.011993, 0.009167, 0.006248, 0.008014, 0.00865, 0.006176],
        [0.01153, 0.009773, 0.006367, 0.008455, 0.009145, 0.006374],
        [0.011896, 0.009225, 0.006814, 0.008319, 0.008764, 0.006564],
        [0.011666, 0.00897, 0.006267, 0.00803, 0.0085, 0.00615],
        [0.011368, 0.009163, 0.005957, 0.00803, 0.008588, 0.005959],
        [0.011085, 0.009443, 0.005976, 0.008062, 0.008743, 0.006027],
        [0.011072, 0.009193, 0.006638, 0.008548, 0.008836, 0.0066],
        [0.011602, 0.00905, 0.005833, 0.00799, 0.008619, 0.005856],
        [0.01244, 0.009117, 0.006814, 0.008386, 0.008833, 0.006632],
        [0.011874, 0.009242, 0.005843, 0.007852, 0.008502, 0.005869],
        [0.011349, 0.009207, 0.00621, 0.008037, 0.008669, 0.006128],
        [0.011422, 0.008271, 0.006067, 0.007585, 0.008062, 0.005868],
        [0.011062, 0.00796, 0.004752, 0.006706, 0.007405, 0.004725],
        [0.011107, 0.008276, 0.005295, 0.007098, 0.007683, 0.005263],
        [0.01091, 0.008195, 0.00541, 0.0072, 0.007717, 0.005366],
        [0.01112, 0.008152, 0.0054, 0.00717, 0.00769, 0.005263],
        [0.010993, 0.007307, 0.00501, 0.006412, 0.006886, 0.004797],
        [0.010937, 0.007689, 0.004933, 0.006685, 0.007224, 0.004825],
        [0.010624, 0.007639, 0.00451, 0.006338, 0.006933, 0.004482],
        [0.010169, 0.008821, 0.00479, 0.007113, 0.007929, 0.004914],
        [0.010332, 0.009127, 0.005543, 0.007785, 0.008443, 0.005641],
        [0.010528, 0.007977, 0.005586, 0.007189, 0.007598, 0.005354],
        [0.010656, 0.007851, 0.005152, 0.006639, 0.007274, 0.004995],
        [0.010928, 0.009048, 0.005333, 0.007562, 0.008338, 0.005418],
        [0.011414, 0.008946, 0.005629, 0.007704, 0.008281, 0.005572],
        [0.011492, 0.008282, 0.005048, 0.007038, 0.007724, 0.005001],
        [0.01091, 0.008123, 0.005086, 0.007113, 0.007712, 0.005007],
        [0.01069, 0.008071, 0.00519, 0.006805, 0.007502, 0.005049],
        [0.010311, 0.007923, 0.004786, 0.006554, 0.007293, 0.004736],
        [0.010774, 0.007794, 0.0048, 0.007061, 0.007612, 0.004977],
        [0.010538, 0.007211, 0.004543, 0.006189, 0.006793, 0.004352],
        [0.010193, 0.007412, 0.004252, 0.006079, 0.006833, 0.004265],
        [0.010178, 0.008175, 0.0045, 0.006533, 0.007424, 0.00459],
        [0.010176, 0.008385, 0.005262, 0.007142, 0.007857, 0.005239],
        [0.009689, 0.00764, 0.004762, 0.006556, 0.007105, 0.004494],
        [0.009405, 0.007095, 0.004105, 0.005948, 0.006612, 0.003906],
        [0.010011, 0.006248, 0.004152, 0.005548, 0.005948, 0.003723],
        [0.010304, 0.006649, 0.003848, 0.005436, 0.006088, 0.0036],
        [0.009848, 0.007485, 0.004376, 0.006337, 0.006964, 0.004437],
        [0.010184, 0.008096, 0.005119, 0.007088, 0.007724, 0.005085],
        [0.009652, 0.00765, 0.004686, 0.006511, 0.007274, 0.004692],
        [0.009625, 0.008175, 0.004614, 0.006756, 0.007564, 0.004848],
    ],
    dtype=np.float64,
)
