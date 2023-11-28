from mapcalc import calculate_map, calculate_map_range

if __name__ == '__main__':
    ground_truth = {
        'boxes':
            [[100, 100, 101, 101]],

        'labels':
            [2]}

    result_dict = {
        'boxes':
            [[100, 100, 101, 101], [150, 150, 151, 151]],

             'labels': [2, 2],

            'scores': [1,1]}


    # calculates the mAP average for the IOU thresholds 0.05, 0.1, 0.15, ..., 0.90, 0.95.
    print(calculate_map_range(ground_truth, result_dict, 0.05, 0.95, 0.05))

    # 0.6666666666666666
    # 0.6578947368421053
