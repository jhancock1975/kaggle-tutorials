# Experiment with code outside the function, then move it into the function once
# you think it is right

# the following lines are given as a hint to get you started
decoded = decode_predictions(preds, top=1)
print('decoded= %s ' % decoded)

def is_hot_dog(preds_array):
    '''
    inputs:
    preds_array:  array of predictions from pre-trained model

    outputs:
    is_hot_dog_list: a list indicating which predictions show hotdog as the most
likely label
    '''
    decoded = decode_predictions(preds_array, top=1)
    return [cur_arr[0][1]=='hotdog' for cur_arr in decoded]

q_1.check()


def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
    # everything from hotdog images should be predicted as hotdog
    test_data = read_and_prep_images(paths_to_hotdog_images)
    hot_dog_preds = is_hot_dog(model.predict(test_data))
    
    # everything from other images should be predicted as not hotdog
    test_data = read_and_prep_images(paths_to_other_images)
    not_dog_preds = is_hot_dog(model.predict(test_data))
    return (hot_dog_preds.count(True) +
not_dog_preds.count(False))/(len(hot_dog_preds)+len(not_dog_preds))

# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths
# were created in the setup code
my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
print("Fraction correct in small test set: {}".format(my_model_accuracy))

# checks that your function calc_accuracy works correctly
q_2.check()
