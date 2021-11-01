from collections import namedtuple
# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal attention')

OPS_PRIMITIVES = [
    'none',
    'skip_connect',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'rb',
    'lrb',
    'srb',
    # 'acb'
]
ATS_PRIMITIVES = [
    'none',
    'skip_connect',
    'pixel_wise_attention', # PA CANet Arxiv 2020 
    'channel_wise_attention', # RCAN CA  √
    'contrast_aware_channel_attention', # IMDN CCA  √
    'spatial_attention',  # from CBAM_ECCV2018  √
    'spatial_attention_v2',  # from BAM_BMVC2018
    'esab', # RFANet_CVPR2020 √
    'cea' # MAFFSRN √
]

COMPACT_PRIMITIVES_UPSAMPLING = [
    'sub_pixel',
    'deconvolution',
    'bilinear',
    'nearest',
    'area',
]

# 搜索到的网络结构 每个结点有两个输入
# 输入格式为(operation name, input node name)
# DARTS_found = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1),
#                                ('dil_conv_3x3', 2), ('sep_conv_3x3', 0),
#                                ('dil_conv_3x3', 2), ('dil_conv_5x5', 3),
#                                ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)],
#                        normal_concat=range(2, 6),
#                        reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0),
#                                ('dil_conv_5x5', 2), ('sep_conv_5x5', 1),
#                                ('dil_conv_5x5', 2), ('dil_conv_5x5', 3),
#                                ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)],
#                        reduce_concat=range(2, 6))
#
# DARTS = DARTS_found
# SRv1 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('lcabv1', 1), ('dil_conv_3x3', 2),
#                         ('rcab', 1), ('dil_conv_3x3', 0), ('lcabv1', 1), ('dil_conv_3x3', 2)],
#                 normal_concat=range(2, 6))
# SRv2 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('lcabv1', 1),
#                         ('rcab', 1), ('lcabv2', 2), ('lcabv1', 1), ('dil_conv_3x3', 3)],
#                 normal_concat=range(2, 6))
# SRv3 = Genotype(normal=[('lcabv2', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('lcabv1', 1),
#                         ('rcab', 1), ('lcabv2', 2), ('dil_conv_3x3', 3), ('lcabv1', 1)],
#                 normal_concat=range(2, 6))
# mycode6.0/experiment/2020-09-08-21-41-37


# [Step: 126000]
SR_x2_v1 = Genotype(normal=[('lrb', 0), ('rb', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('lrb', 1), ('rb', 2)],
                    attention=[('pa_sa_cascade', 0), ('pa_sa_cascade', 1), ('sa_ca_dual', 2), ('ca_sa_cascade', 3)])

SR_x2_end = Genotype(normal=[('lrb', 0), ('rb', 0), ('rb', 1), ('sep_conv_3x3', 0), ('lrb', 1), ('rb', 2)],
                     attention=[('ca_sa_cascade', 0), ('pa_sa_cascade', 1), ('sa_ca_dual', 2), ('ca_sa_cascade', 3)])
# PC_DARTS_SR_x2_end = Genotype(normal=[('lrb', 0), ('dil_conv_3x3', 1), ('rb', 0), ('lrb', 1), ('rb', 0), ('acb', 1), ('rb', 4), ('rb', 1)],
#                               attention=[('esab', 0), ('cea', 0), ('cea', 1), ('esab', 0), ('cea', 1), ('cea', 2)])
SR_x2_final = Genotype(normal=[('rb', 0), ('lrb', 0), ('rb', 1), ('rb', 0), ('srb', 1), ('rb', 2)], 
attention=[('esab', 0), ('cea', 0), ('cea', 1), ('esab', 0), ('skip_connect', 1), ('cea', 2)])
