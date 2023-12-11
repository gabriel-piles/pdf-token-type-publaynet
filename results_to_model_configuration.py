if __name__ == '__main__':
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