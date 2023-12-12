def results():
    options = "f1	context_size	num_boost_round	num_leaves	bagging_fraction	lambda_l1	lambda_l2	feature_fraction	bagging_freq	min_data_in_leaf	feature_pre_filter	boosting_type	objective	metric	learning_rate	seed	num_class	verbose	deterministic	resume_training	early_stopping_rounds"
    model_configuration_line = "0.994	4	652	427	0.538576362137786	0.0005757631630210714	0.014421883568195633	0.5093595305684682	4	91	False	gbdt	multiclass	multi_logloss	0.1	22	13	-1	False	False"

    print("configuration_dict = dict()")

    for option, value in zip(options.split(), model_configuration_line.split()):
        if option == "f1":
            continue

        value = f'"{value}"' if value in ["gbdt", "multiclass", "multi_logloss"] else value
        print(f'configuration_dict["{option}"]', '=', value)

    print()
    print("model_configuration = ModelConfiguration(**configuration_dict)")


def score_without_figure():
    scores_text = ['''1	
VGT
0.962	0.950	0.939	0.968	0.981	0.971''',
                   '''2	
VSR
0.957	0.967	0.931	0.947	0.974	0.964		
''',
                   '''3	
DETR
0.957	0.947	0.918	0.964	0.981	0.975''',
                   '''4	
LayoutLMv3-B
0.951	0.945	0.906	0.955	0.979	0.970''',
                   '''5	
DiT-L
0.949	0.944	0.893	0.960	0.978	0.972''',
        '''6	
UDoc
0.939	0.939	0.885	0.937	0.973	0.964''',
        '''7	
ResNext-101-32×8d
0.935	0.930	0.862	0.940	0.976	0.968''',
                   '''8	
DeiT-B
0.932	0.934	0.874	0.921	0.972	0.957
''',
                   '''9	
BEiT-B
0.931	0.934	0.866	0.924	0.973	0.957''',
                   '''10	
Mask-RCNN
0.910	0.916	0.840	0.886	0.960	0.949''',
                   '''11	
Faster-RCNN
0.902	0.910	0.826	0.883	0.954	0.937		
'''
                   ]
    print('\t'.join('''Rank	Model	
Overall
Text	
Title	
List	
Table	
Figure'''.split()))
    for score_text in scores_text:
        values = list(score_text.split())
        scores = [float(x) for x in values[3:7]]

        values[2] = round(sum(scores)/4, 3)
        print("\t".join([str(x) for x in values]))


if __name__ == '__main__':
    score_without_figure()