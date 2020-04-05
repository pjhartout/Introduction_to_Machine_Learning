


#python subtask1and2.py --train_features ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_train.csv --train_labels ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_train_label.csv --test_features ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_test.csv --predictions ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/predictions_subtask3.zip --sampling_strategy adasyn --k_fold 5 --scaler standard
rm -rf runs/
# Run subtask 3 on GPU
python subtask3.py --train_features ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_train.csv --train_labels ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_train_label.csv --test_features ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/preprocessed_test.csv --predictions ~/Documents/GitHub/Introduction_to_Machine_Learning/projects/project_2/data/predictions_subtask3.zip --k_fold 5 --scaler minmax --model ANN --epochs 200
