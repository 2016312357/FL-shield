from __future__ import print_function

"""
train attack model for ori examples
"""
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Activation, Conv1D, Conv2D, Flatten
import tensorflow as tf
import numpy as np
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def model_attack(input_shape, num_classes=25):
    inputs = Input(shape=input_shape)
    # x = Dense(256, activation='relu')(inputs)
    x = Dense(10, activation='relu')(inputs)
    # x = Dense(int(input_shape[0]/4), activation='relu')(x)
    # x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def model_attack1(input_shape, num_classes=25):
    inputs = Input(shape=input_shape)
    # x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(inputs)
    # x = Dense(int(input_shape[0]/4), activation='relu')(x)
    # x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def model_attack2(input_shape, num_classes=25):
    # print(input_shape)
    inputs = Input(shape=input_shape)
    dim1 = input_shape[0]
    dim2 = input_shape[1]
    x = Dropout(0.2)(inputs)
    x = Conv2D(50,
               kernel_size=(1, dim2),
               strides=(1, 1),
               padding='valid',
               activation='relu',
               data_format="channels_last",
               # input_shape=(dim1, dim2, 1,),
               )(x),
    x = Flatten()(x[0])
    x = Dropout(0.2)(x),
    x = Dense(
        512,
        activation='relu',
        bias_initializer='zeros')(x[0]),
    x = Dense(
        256,
        activation='relu',
        bias_initializer='zeros')(x[0]),
    output = Dense(num_classes, activation='softmax')(x[0])
    model = Model(inputs=inputs, outputs=output)
    return model


keraslayers = keras.layers


def cnn_for_fcn_gradients(input_shape):
    """
    Creates a CNN submodule for Dense layer gradients.
    """
    # Input container
    dim1 = int(input_shape[0])
    dim2 = int(input_shape[1])
    initializer = tf.keras.initializers.random_normal(0.0, 0.01)

    # CNN module
    cnn = tf.keras.Sequential(
        [
            keraslayers.Dropout(0.2, input_shape=(dim1, dim2, 1,), ),
            keraslayers.Conv2D(
                100,
                kernel_size=(1, dim2),
                strides=(1, 1),
                padding='valid',
                activation=tf.nn.relu,
                data_format="channels_last",
                # input_shape=(dim1, dim2, 1,),
                kernel_initializer=initializer,
                bias_initializer='zeros',
            ),
            keraslayers.Flatten(),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                2024,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dropout(0.2, input_shape=(dim1, dim2, 1,), ),
            keraslayers.Dense(
                512,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                256,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
        ]
    )
    return cnn


def cnn_for_cnn_gradients(input_shape):
    """
    Creates a CNN submodule for Conv layer gradients
    """
    dim1 = int(input_shape[3])
    dim2 = int(input_shape[0])
    dim3 = int(input_shape[1])
    dim4 = int(input_shape[2])
    initializer = tf.keras.initializers.random_normal(0.0, 0.01)
    # CNN module
    cnn = tf.keras.Sequential(
        [
            keraslayers.Conv2D(
                dim1,
                kernel_size=(dim2, dim3),
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                input_shape=(dim1, dim2, dim4),
                kernel_initializer=initializer,
                bias_initializer='zeros',
                name='cnn_grad_layer'
            ),
            keraslayers.Flatten(name='flatten_layer'),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )

        ]
    )
    return cnn


prop = {'21': 0, '408': 1, '335': 2, '529': 3, '750': 3, '730': 4, '838': 1, '188': 1, '304': 1, '364': 5, '695': 0,
        '679': 1, '315': 6, '606': 7, '358': 6, '460': 8, '922': 3, '924': 8, '751': 8, '787': 1, '826': 9, '411': 6,
        '283': 7, '508': 10, '159': 1, '393': 1, '279': 7, '649': 1, '572': 6, '486': 6, '792': 7, '812': 11, '462': 1,
        '582': 1, '185': 12, '425': 1, '743': 7, '290': 5, '235': 6, '759': 1, '938': 11, '11': 8, '912': 8, '435': 5,
        '638': 5, '140': 1, '678': 6, '799': 3, '314': 1, '203': 1, '111': 5, '878': 6, '82': 7, '657': 13, '617': 0,
        '244': 11, '800': 7, '8': 3, '100': 2, '570': 6, '881': 10, '666': 3, '306': 8, '629': 8, '613': 10, '534': 1,
        '485': 6, '864': 7, '136': 8, '725': 1, '650': 5, '515': 10, '75': 14, '717': 11, '870': 1, '791': 6, '906': 12,
        '105': 5, '223': 1, '334': 12, '217': 8, '440': 8, '610': 1, '135': 1, '863': 1, '42': 3, '297': 6, '811': 6,
        '296': 3, '55': 7, '164': 15, '727': 1, '598': 10, '276': 1, '352': 7, '263': 7, '240': 6, '909': 6, '400': 3,
        '654': 1, '711': 1, '639': 12, '288': 10, '332': 1, '127': 13, '121': 12, '163': 3, '278': 12, '524': 6,
        '173': 8, '919': 8, '153': 1, '94': 1, '601': 9, '555': 6, '648': 5, '814': 8, '503': 0, '99': 1, '507': 0,
        '865': 9, '829': 0, '729': 1, '588': 1, '859': 8, '755': 6, '25': 5, '76': 1, '95': 3, '313': 10, '824': 8,
        '677': 8, '754': 12, '370': 0, '54': 2, '577': 1, '427': 16, '833': 0, '83': 8, '168': 8, '542': 1, '291': 1,
        '891': 3, '474': 2, '795': 7, '698': 7, '935': 16, '704': 12, '383': 3, '831': 8, '194': 3, '624': 1, '558': 0,
        '722': 17, '233': 5, '113': 2, '867': 4, '158': 6, '437': 8, '12': 8, '272': 4, '69': 5, '337': 4, '43': 12,
        '165': 8, '576': 2, '882': 5, '452': 3, '705': 1, '249': 1, '373': 8, '686': 6, '177': 7, '19': 12, '660': 1,
        '329': 6, '763': 4, '546': 2, '858': 6, '232': 4, '354': 12, '673': 6, '68': 1, '716': 3, '500': 3, '67': 1,
        '696': 8, '108': 6, '225': 3, '850': 11, '637': 8, '726': 3, '245': 1, '663': 8, '343': 5, '138': 16, '890': 1,
        '404': 7, '562': 3, '830': 7, '182': 7, '707': 12, '847': 1, '775': 2, '200': 7, '456': 11, '381': 9, '941': 1,
        '651': 18, '449': 12, '764': 6, '644': 18, '106': 18, '192': 6, '720': 3, '662': 12, '467': 5, '15': 6,
        '418': 13, '311': 11, '936': 8, '806': 10, '740': 6, '586': 1, '322': 1, '262': 1, '115': 5, '473': 1, '804': 6,
        '123': 9, '9': 1, '30': 1, '414': 7, '550': 1, '510': 8, '166': 6, '4': 11, '230': 1, '259': 1, '359': 1,
        '399': 8, '132': 8, '646': 1, '380': 5, '66': 1, '513': 3, '908': 12, '612': 6, '154': 1, '137': 6, '195': 4,
        '856': 10, '671': 7, '134': 7, '926': 14, '839': 14, '921': 1, '204': 12, '61': 5, '589': 19, '338': 12,
        '78': 3, '461': 1, '539': 3, '801': 0, '615': 6, '537': 5, '413': 6, '665': 3, '658': 7, '544': 8, '27': 12,
        '239': 9, '234': 18, '118': 3, '628': 13, '434': 1, '708': 17, '24': 9, '672': 3, '635': 8, '471': 1, '506': 7,
        '895': 12, '207': 10, '560': 1, '516': 12, '453': 1, '13': 6, '5': 8, '110': 1, '832': 11, '465': 8, '884': 5,
        '809': 10, '880': 1, '808': 20, '643': 4, '305': 7, '18': 8, '470': 7, '128': 10, '73': 1, '769': 2, '378': 1,
        '351': 6, '528': 1, '871': 2, '227': 2, '241': 1, '209': 6, '303': 1, '327': 1, '406': 6, '62': 3, '226': 1,
        '526': 10, '346': 8, '552': 8, '533': 12, '641': 1, '721': 14, '307': 1, '40': 4, '683': 12, '608': 8, '199': 0,
        '369': 1, '35': 17, '901': 2, '519': 8, '340': 5, '180': 3, '772': 0, '914': 8, '193': 1, '339': 19, '848': 5,
        '816': 8, '735': 15, '710': 1, '581': 8, '362': 17, '109': 8, '44': 11, '379': 7, '927': 7, '517': 1, '237': 3,
        '931': 6, '653': 2, '255': 14, '760': 8, '585': 12, '874': 4, '181': 2, '868': 7, '488': 11, '547': 6, '310': 6,
        '152': 6, '902': 9, '888': 4, '690': 20, '184': 12, '396': 5, '699': 8, '424': 10, '390': 0, '50': 0, '464': 0,
        '189': 9, '512': 8, '502': 1, '361': 1, '367': 1, '490': 9, '875': 1, '363': 1, '360': 8, '655': 15, '647': 6,
        '407': 5, '521': 1, '238': 3, '640': 1, '374': 2, '28': 0, '145': 14, '328': 3, '450': 6, '796': 0, '899': 8,
        '156': 6, '557': 0, '187': 6, '659': 6, '688': 3, '466': 1, '481': 18, '285': 7, '543': 4, '201': 0, '349': 18,
        '694': 7, '142': 8, '316': 8, '845': 16, '574': 6, '231': 12, '786': 5, '556': 6, '122': 0, '807': 15, '747': 8,
        '752': 18, '265': 2, '706': 1, '821': 5, '139': 1, '514': 7, '525': 3, '893': 1, '198': 1, '761': 1, '564': 18,
        '595': 7, '469': 6, '756': 13, '131': 3, '212': 6, '229': 12, '487': 5, '197': 11, '877': 8, '757': 1, '632': 1,
        '420': 6, '518': 0, '857': 3, '813': 1, '293': 0, '468': 5, '257': 1, '737': 7, '536': 5, '784': 3, '308': 18,
        '621': 1, '218': 3, '520': 15, '841': 16, '900': 18, '527': 12, '684': 1, '904': 1, '17': 7, '457': 20,
        '693': 15, '300': 7, '736': 0, '876': 8, '186': 2, '923': 1, '723': 2, '281': 1, '151': 3, '869': 1, '444': 19,
        '548': 0, '770': 1, '573': 18, '148': 5, '3': 0, '179': 14, '709': 8, '319': 7, '823': 9, '625': 7, '439': 3,
        '90': 6, '172': 10, '773': 1, '701': 12, '325': 11, '538': 4, '872': 1, '676': 7, '116': 15, '146': 9, '87': 3,
        '208': 5, '853': 0, '491': 0, '431': 10, '274': 1, '607': 15, '77': 11, '376': 8, '545': 11, '162': 9, '86': 3,
        '910': 15, '222': 7, '252': 5, '183': 4, '246': 1, '915': 14, '210': 5, '2': 8, '273': 8, '892': 8, '798': 0,
        '405': 15, '416': 1, '202': 6, '642': 1, '930': 4, '7': 3, '366': 1, '26': 5, '286': 1, '587': 8, '779': 1,
        '81': 1, '782': 9, '685': 12, '120': 8, '484': 1, '243': 6, '347': 1, '603': 7, '530': 5, '907': 8, '268': 5,
        '604': 6, '929': 4, '397': 1, '157': 5, '321': 6, '84': 2, '631': 1, '822': 12, '889': 11, '72': 3, '228': 1,
        '260': 9, '224': 6, '52': 1, '124': 1, '60': 15, '377': 1, '130': 13, '33': 1, '836': 9, '58': 7, '509': 3,
        '592': 1, '270': 1, '46': 10, '768': 3, '430': 4, '59': 6, '149': 10, '645': 7, '531': 20, '911': 0, '781': 1,
        '549': 4, '103': 1, '602': 8, '785': 5, '554': 4, '169': 8, '387': 14, '170': 15, '575': 10, '504': 0, '415': 6,
        '600': 7, '6': 2, '501': 1, '897': 8, '211': 20, '88': 12, '391': 1, '475': 7, '819': 3, '330': 6, '357': 2,
        '92': 14, '331': 14, '652': 8, '216': 5, '627': 5, '886': 1, '789': 8, '778': 1, '38': 8, '540': 5, '563': 12,
        '866': 8, '565': 1, '634': 5, '365': 19, '618': 1, '353': 4, '112': 20, '852': 3, '117': 1, '734': 8, '14': 4,
        '63': 10, '887': 1, '248': 1, '451': 1, '441': 11, '916': 5, '762': 3, '905': 8, '616': 4, '394': 3, '499': 7,
        '478': 8, '566': 1, '718': 11, '389': 0, '805': 8, '879': 3, '355': 1, '689': 8, '675': 8, '356': 17, '39': 14,
        '843': 12, '336': 20, '854': 1, '119': 7, '571': 9, '292': 7, '1': 11, '312': 8, '674': 1, '724': 2, '454': 8,
        '323': 1, '22': 0, '594': 6, '448': 14, '174': 3, '401': 15, '428': 1, '939': 1, '597': 8, '765': 1, '167': 8,
        '326': 3, '633': 7, '496': 1, '220': 12, '593': 6, '410': 9, '41': 5, '568': 6, '846': 19, '738': 11, '661': 7,
        '51': 6, '155': 8, '827': 5, '47': 10, '350': 1, '849': 1, '409': 3, '745': 0, '932': 6, '175': 4, '731': 6,
        '623': 6, '271': 5, '385': 0, '426': 6, '446': 6, '680': 19, '442': 1, '388': 8, '569': 6, '160': 7, '395': 8,
        '302': 6, '309': 4, '384': 7, '896': 0, '837': 9, '873': 3, '739': 11, '788': 3, '392': 0, '920': 9, '820': 1,
        '614': 6, '429': 1, '828': 12, '421': 7, '289': 13, '342': 8, '20': 17, '803': 3, '219': 7, '114': 7, '126': 19,
        '766': 8, '45': 7, '494': 3, '371': 5, '459': 1, '266': 3, '445': 0, '89': 3, '532': 1, '862': 2, '53': 7,
        '783': 10, '261': 3, '299': 16, '928': 1, '917': 1, '301': 1, '774': 1, '250': 2, '104': 1, '97': 9, '777': 7,
        '318': 18, '443': 20, '458': 11, '93': 2, '692': 5, '433': 9, '728': 2, '605': 5, '317': 3, '489': 8, '258': 1,
        '375': 14, '254': 6, '498': 0, '622': 7, '497': 1, '178': 8, '269': 12, '275': 5, '71': 4, '147': 12, '125': 19,
        '682': 7, '144': 7, '732': 8, '205': 19, '934': 5, '733': 8, '482': 1, '295': 6, '561': 5, '477': 1, '386': 20,
        '79': 3, '840': 9, '70': 5, '472': 1, '324': 1, '818': 12, '344': 12, '700': 1, '422': 14, '37': 1, '247': 5,
        '636': 6, '749': 8, '505': 8, '535': 6, '670': 11, '214': 12, '483': 4, '898': 17, '742': 1, '855': 12,
        '280': 12, '860': 18, '85': 6, '236': 0, '402': 5, '284': 2, '476': 1, '48': 3, '746': 5, '463': 15, '834': 8,
        '817': 1, '714': 5, '802': 3, '669': 8, '599': 1, '579': 6, '940': 3, '681': 10, '206': 1, '744': 10, '16': 14,
        '748': 3, '767': 5, '368': 1, '57': 13, '591': 12, '559': 2, '551': 7, '287': 20, '264': 0, '656': 6, '438': 3,
        '96': 9, '578': 3, '611': 12, '851': 8, '691': 6, '190': 3, '256': 13, '780': 7, '703': 6, '107': 4, '23': 9,
        '423': 8, '553': 6, '447': 3, '522': 5, '176': 4, '815': 8, '102': 7, '933': 1, '101': 1, '702': 8, '942': 12,
        '221': 1, '913': 1, '583': 5, '793': 1, '56': 12, '590': 6, '31': 9, '580': 1, '348': 1, '143': 11, '626': 4,
        '797': 8, '810': 8, '495': 5, '835': 2, '943': 1, '282': 3, '196': 0, '91': 10, '492': 6, '34': 3, '479': 6,
        '790': 11, '687': 15, '251': 16, '596': 9, '455': 3, '776': 12, '620': 0, '417': 8, '771': 1, '65': 6, '10': 19,
        '253': 12, '333': 8, '215': 7, '511': 1, '372': 1, '133': 5, '667': 12, '141': 7, '267': 5, '49': 1, '758': 1,
        '80': 3, '36': 1, '794': 6, '713': 8, '191': 3, '493': 5, '541': 1, '242': 6, '382': 5, '741': 0, '619': 1,
        '161': 19, '885': 8, '925': 20, '697': 8, '345': 12, '668': 0, '419': 19, '894': 6, '861': 1, '412': 6, '74': 4,
        '825': 5, '844': 5, '32': 1, '29': 7, '883': 12, '753': 20, '298': 2, '320': 1, '436': 3, '719': 8, '150': 9,
        '715': 11, '98': 2, '609': 1, '712': 1, '584': 1, '567': 14, '664': 5, '129': 10, '903': 6, '630': 15, '918': 4,
        '277': 3, '842': 0, '171': 6, '398': 8, '213': 2, '432': 14, '294': 11, '937': 6, '403': 8, '523': 3, '64': 6,
        '341': 1, '480': 21}

# 用梯度攻击！！！！！！！！！
random.seed(0)

# f = os.listdir('./checkpoints-2layer-gradients/')
# f = np.load('ml-100k-clean.npz', allow_pickle=True)
#name = 'ml-100k-clean.npz'
name='ml-100k-layer_1_epoch-500-gc0.01.npz'
#name='ml-10k-layer_1_epoch-1000.npz'
#name = 'ml-100k-layer_0_1_3_epoch-500-dp.npz'
#f = np.load('ml-100k-lap-5.npz', allow_pickle=True)
f = np.load(name, allow_pickle=True)
# random.shuffle(f)
# print(f)
attack_data = []
labels = []
# print(f['arr_1'])

print('label', f['arr_1'].shape)  # 147，10
for weights, lab in zip(f['arr_0'], f['arr_1']):  # (38, 10, 8)
    # print('user group',lab)
    for weight, la in zip(weights, lab):  # 参与不同epoch的用户组
        # print('user id in one group', la)
        labels.append(prop[str(la)])
        d = weight[0].reshape(1, -1)
        for layerid in range(1, len(weight)):
            d = np.concatenate((d, weight[layerid].reshape(1, -1)), axis=-1)
        # print(d.shape)
        attack_data.append(d)
        # attack_data.append(np.expand_dims(d, axis=0))  # .reshape((1,-1))
        # attack_dat=np.array(attack_data)

'''
if int(ff[1][:-4])>200:
        pass
        (933, 10)
        (1683, 10)
        (933, 5)
        (1683, 5)
        (20, 64)(64,)
        (64, 1)(1,)   
    else:'''

label = np.asarray(labels)
attack_data = np.vstack(attack_data)  # .reshape((attack_data.shape,1))

print('all data shape', label.shape, attack_data.shape)  # (380,) (380, 933, 10)
# 21 class
y_test = np.zeros([len(attack_data), 21])
# one-hot encoding
for i in range(len(y_test)):
    y_test[i, label[i]] = 1.
count = [0 for i in range(21)]
for i in label:
    count[i] += 1
print('label count', count)  # [21, 90, 14, 36, 0, 45, 19, 21, 44, 4, 8, 21, 29, 2, 10, 2, 1, 0, 5, 7, 1]


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(attack_data, y_test, test_size=0.3, random_state=0)


input_shape = train_x.shape[1:]


model = model_attack(input_shape=input_shape, num_classes=train_y.shape[1])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

'''model.set_weights(np.load("./attack_checkpoint/attack_model_ganju.npz",allow_pickle=True)['x'])  # 1; 1280,
scores = model.evaluate(attack_data, y_test, batch_size=len(y_test), verbose=0)
print("Ganju attack, initial Test loss: {},Test accuracy: {}".format(scores[0], scores[1]))'''

batch_size = 16

benign_index_array = np.arange(train_x.shape[0])
batch_num = np.int(np.ceil(train_x.shape[0] / batch_size))


epochs = 100
for i in range(epochs):
    np.random.shuffle(benign_index_array)
    for j in np.arange(batch_num):
        x_batch = train_x[
            benign_index_array[(j % batch_num) * batch_size:min((j % batch_num + 1) * batch_size, train_x.shape[0])]]
        y_batch = train_y[
            benign_index_array[(j % batch_num) * batch_size:min((j % batch_num + 1) * batch_size, train_x.shape[0])]]
        model.train_on_batch(x_batch, y_batch)
    if (i + 1) % 5 == 0:
        print("epoch {}".format(i))
        scores = model.evaluate(train_x, train_y, verbose=0)
        print("Tr loss: {},Tr accuracy: {}".format(scores[0], scores[1]))

        scores = model.evaluate(test_x, test_y, verbose=0)
        weights = model.get_weights()
        np.savez("./attack_checkpoint/attack_model_ganju-1.npz", x=weights)  # 1; 1280,
        print("Saving, Test loss: {},Test accuracy: {}".format(scores[0], scores[1]))


'''model.fit(train_x, train_y, epochs=100, batch_size=16, verbose=1, shuffle=True)
scores = model.evaluate(test_x, test_y, verbose=0)
print("Test loss: {},Test accuracy: {}".format(scores[0], scores[1]))
'''
'''
0
Tr loss: 0.003515384900521718,Tr accuracy: 1.0
Test loss: 1.7177974780400593,Test accuracy: 0.8301886743719473
1
Tr loss: 0.0024311896877970817,Tr accuracy: 1.0
Test loss: 1.4657359678040511,Test accuracy: 0.8490565989002492
2
Tr loss: 0.06578854471445084,Tr accuracy: 0.9919137466307277
Test loss: 2.1095090932066336,Test accuracy: 0.7484276748303348

Tr loss: 0.0707675214766492,Tr accuracy: 0.9973045822102425
Test loss: 2.1359705580105572,Test accuracy: 0.7044025175976303
3

Tr loss: 5.317713318576514e-05,Tr accuracy: 1.0
Test loss: 1.4718324403342962,Test accuracy: 0.8553459070763498

layerid 4
Tr loss: 0.05913224011016985,Tr accuracy: 1.0
Test loss: 2.9611027255748055,Test accuracy: 0.5849056626265904

5


6
Tr loss: 1.9526674252636027,Tr accuracy: 0.29380053916388765
Test loss: 2.647600788740242,Test accuracy: 0.2264150932150067


'''

'''for k in range(21):
    f = os.listdir('./checkpoints-2layer/')
        file = f[0]
        weight = np.load('./checkpoints-2layer/' + file)['arr_0']
        print('loading model', weight[-4:-3][0].shape)

    #adv = np.load('adv-10-%d.npz' % k)['arr_0'][-4]  # './checkpoints-2layer/' + file)['arr_0']
    #print(k, 'test adv label: ', np.argmax(model.predict(adv.reshape((1,-1))), axis=-1)[0])
    try:
        adv = np.load('adv-%d.npz' % k)['arr_0'][-4].reshape((1,-1))  # './checkpoints-2layer/' + file)['arr_0']
        print(k, 'transfer test adv label: ', np.argmax(model.predict(adv), axis=-1)[0])
    except:
        pass
'''
# only test
