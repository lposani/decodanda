from all_dichotomies import test_all_dichotomies
from balancing_confounds import test_balancing_confounds
from non_linear_classifier import test_non_linear_decoder
from divide_into_conditions import test_divide_into_conditions

if __name__ == '__main__':
    try:
        test_divide_into_conditions()
        print("Test divide into conditions:\tpassed")
    except AssertionError:
        print("X\tTest divide into conditions:\tFAILED\tX")
    try:
        test_non_linear_decoder()
        print("Test non linear classifier:\tpassed")
    except AssertionError:
        print("X\tTest non linear classifier:\tFAILED\tX")
    try:
        test_balancing_confounds()
        print("Test balancing confounds:\tpassed")
    except AssertionError:
        print("X\tTest balancing confounds:\tFAILED\tX")
    try:
        test_all_dichotomies()
        print("Test all dichotomies:\tpassed")
    except AssertionError:
        print("X\tTest all dichotomies:\tFAILED\tX")
    print("\nAll tests passed.")



