import argparse
import logging

import pandas as pd
import numpy as np

TYPICAL_VALUES = {
    "pid": 15788.831218741774,
    "Time": 7.014398525927875,
    "Age": 62.07380889707818,
    "EtCO2": 32.88311356434632,
    "PTT": 40.09130983590656,
    "BUN": 23.192663516538175,
    "Lactate": 2.8597155076236422,
    "Temp": 36.852135856500034,
    "Hgb": 10.628207669881103,
    "HCO3": 23.488100167210746,
    "BaseExcess": -1.2392844571830848,
    "RRate": 18.154043187688046,
    "Fibrinogen": 262.496911351785,
    "Phosphate": 3.612519413287318,
    "WBC": 11.738648535345682,
    "Creatinine": 1.4957773156474896,
    "PaCO2": 41.11569643111729,
    "AST": 193.4448880402708,
    "FiO2": 0.7016656642357807,
    "Platelets": 204.66642639312448,
    "SaO2": 93.010527124635,
    "Glucose": 142.169406624713,
    "ABPm": 82.11727559995713,
    "Magnesium": 2.004148832962384,
    "Potassium": 4.152729193815373,
    "ABPd": 64.01471072970384,
    "Calcium": 7.161149186763874,
    "Alkalinephos": 97.79616327960757,
    "SpO2": 97.6634493216935,
    "Bilirubin_direct": 1.390723226703758,
    "Chloride": 106.26018538478121,
    "Hct": 31.28308971681893,
    "Heartrate": 84.52237068276303,
    "Bilirubin_total": 1.6409406684190786,
    "TroponinI": 7.269239936440605,
    "ABPs": 122.3698773806418,
    "pH": 7.367231494050988,
}

IDENTIFIERS = ["pid", "Time"]
MEDICAL_TESTS = [
    "LABEL_BaseExcess",
    "LABEL_Fibrinogen",
    "LABEL_AST",
    "LABEL_Alkalinephos",
    "LABEL_Bilirubin_total",
    "LABEL_Lactate",
    "LABEL_TroponinI",
    "LABEL_SaO2",
    "LABEL_Bilirubin_direct",
    "LABEL_EtCO2",
]
VITAL_SIGNS = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
SEPSIS = ["LABEL_Sepsis"]


def load_data():
    """Loads data to three different dataframes.

    Returns:
        df_train, df_train_label, df_test (pandas.core.frame.DataFrame): three dataframes containing
        the training features, training labels and testing features respectively.

    """
    if FLAGS.nb_of_patients is not None:
        rows_to_load = (FLAGS.nb_of_patients * 12) + 1
    else:
        rows_to_load = None
    df_train = pd.read_csv(FLAGS.train_features, nrows=rows_to_load)
    df_train_label = pd.read_csv(FLAGS.train_labels, nrows=rows_to_load)
    df_test = pd.read_csv(FLAGS.test_features, nrows=rows_to_load)
    return df_train, df_train_label, df_test


# slower version - supports patient specific mean
def fill_na_with_average_patient_column(df, logger):
    """Fills NaNs with the average value of each column for each patient if available,
    otherwise column-wide entry

    Args:
        df (pandas.core.frame.DataFrame): data to be transformed
        logger (Logger): logger

    Returns:
        df (pandas.core.frame.DataFrame): dataframe containing the transformed data
    """
    columns = list(df.columns)
    for i, column in enumerate(columns):
        logger.info(
            "{} column of {} columns processed".format(i + 1, len(columns))
        )
        # Fill na with patient average
        df[[column]] = df.groupby(["pid"])[column].transform(
            lambda x: x.fillna(x.mean())
        )

    # Fill na with overall column average for lack of a better option for now
    df = df.fillna(df.mean())
    if df.isnull().values.any():
        columns_with_na = df.columns[df.isna().any()].tolist()
        for column in columns_with_na:
            df[column] = TYPICAL_VALUES[column]
    return df


def fill_na_with_average_column(df):
    """Quick version of fill_na_with_average_patient_column - does not support patient average
    and results in loss of information.

    Note:
        Inserted dict with typical values as global var because running the script on parts of the
        data leads to errors associated with NaNs because there is not a single sample.

    Args:
        df (pandas.core.DataFrame): data to be transformed

    Returns:
        df (pandas.core.frame.DataFrame): dataframe containing the transformed data
    """

    df = df.fillna(df.mean(numeric_only=True))
    if df.isnull().values.any():
        columns_with_na = df.columns[df.isna().any()].tolist()
        for column in columns_with_na:
            df[column] = TYPICAL_VALUES[column]
    return df


def main(logger):
    """Primary function reading, preprocessing and modelling the data

    Args:
        logger (Logger): logger to get information about the status of the script when running

    Returns:
        None
    """

    logger.info("Loading data")
    df_train, df_train_label, df_test = load_data()
    logger.info("Finished Loading data")

    logger.info("Preprocess training set")
    # Would be useful to distribute/multithread this part
    df_train_preprocessed = fill_na_with_average_patient_column(
        df_train, logger
    )

    # Cast training labels for these tasks
    df_train_label[MEDICAL_TESTS + VITAL_SIGNS + SEPSIS] = df_train_label[
        MEDICAL_TESTS + VITAL_SIGNS + SEPSIS
    ].astype(int)
    # Merging pids to make sure they map correctly.
    df_train_preprocessed_merged = pd.merge(
        df_train_preprocessed,
        df_train_label,
        how="left",
        left_on="pid",
        right_on="pid",
    )

    df_train_label_preprocessed = df_train_preprocessed_merged[IDENTIFIERS + MEDICAL_TESTS + VITAL_SIGNS+ SEPSIS ]
    logger.info("Preprocess test set")
    df_test_preprocessed = fill_na_with_average_patient_column(df_test, logger)

    logger.info("Export preprocessed train and test set")
    # Export pandas dataframe to csv.
    df_train_label_preprocessed.to_csv(
        FLAGS.preprocess_label, index=False, float_format="%.3f"
    )
    df_train_preprocessed.to_csv(
        FLAGS.preprocess_train, index=False, float_format="%.3f"
    )
    df_test_preprocessed.to_csv(
        FLAGS.preprocess_test, index=False, float_format="%.3f"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI args for folder and file \
    directories"
    )

    parser.add_argument(
        "--train_features",
        "-train_f",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
                        features",
    )

    parser.add_argument(
        "--test_features",
        "-test",
        type=str,
        required=True,
        help="path to the CSV file containing the testing \
                            features",
    )

    parser.add_argument(
        "--train_labels",
        "-train_l",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
                                labels",
    )

    parser.add_argument(
        "--preprocess_train",
        "-pre_train",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
                                 preprocessed data",
    )

    parser.add_argument(
        "--preprocess_test",
        "-pre_test",
        type=str,
        required=True,
        help="path to the CSV file containing the test \
                                preprocessed data",
    )

    parser.add_argument(
        "--nb_of_patients",
        "-nb_pat",
        type=int,
        required=False,
        help="Number of patients to consider in run. If not specified, then consider all patients",
    )

    parser.add_argument(
        "--preprocess_label",
        "-pre_label",
        type=str,
        required=True,
        help="path to the CSV file containing the training \
                                label data",
    )



   
    FLAGS = parser.parse_args()

    # clear logger.
    logging.basicConfig(
        level=logging.DEBUG, filename="script_status_subtask1and2.log"
    )

    logger = logging.getLogger("IML-P2-T1T2")

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] "
            "%(message)s"
        )
    )
    logger.addHandler(stream_handler)

    main(logger)
