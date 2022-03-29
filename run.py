import argparse
from models.resampling import OversamplingResampler
from preprocess import GlobalDictionary
from utils.out import Out
from misc import build_atomic_features
from preprocess.java_parser.javafile import JavaFile
from preprocess.java_parser.ast2seq import AST2Seq
import preprocess.java_parser.tokenizers
import preprocess.java_parser.ast2seq
import preprocess.java_parser.javafile
import preprocess.java_parser.ast2seq
import preprocess.dataset
from models import LSTMModel, TFIDFModel, DBNModel, CNNModel
from utils.misc import merge_dict
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from explanation.tokenomission import TokenOmission
from explanation.metrics import deletion_metrics, explanation_precision
from explanation.feature_representation import Instance, features_to_tokens
from explanation.randomselection import RandomSelection
from explanation.lime import LIME


def get_project_name(dataset_name):
    splited = dataset_name.split("-")
    return splited[0] + "-" + ".".join(splited[1]), \
           splited[0] + "-" + ".".join(splited[2])


project_mapper = {"xalan": "xalan-250-260", "poi": "poi-150-200", "lucene": "lucene-200-220"}

# Specify the nodes captured from the AST.
nodes = [
    # Invocation nodes.
    preprocess.java_parser.tokenizers.MethodInvocation,
    preprocess.java_parser.tokenizers.SuperMethodInvocation,
    preprocess.java_parser.tokenizers.ClassCreator,

    # Declaration nodes.
    preprocess.java_parser.tokenizers.InterfaceDeclaration,
    preprocess.java_parser.tokenizers.EnumDeclaration,
    preprocess.java_parser.tokenizers.ClassDeclaration,
    preprocess.java_parser.tokenizers.ConstructorDeclaration,
    preprocess.java_parser.tokenizers.MethodDeclaration,
    preprocess.java_parser.tokenizers.FormalParameter,

    # Control flow nodes.
    preprocess.java_parser.tokenizers.IfStatement,
    preprocess.java_parser.tokenizers.ForStatement,
    preprocess.java_parser.tokenizers.WhileStatement,
    preprocess.java_parser.tokenizers.DoStatement,
    preprocess.java_parser.tokenizers.AssertStatement,
    preprocess.java_parser.tokenizers.BreakStatement,
    preprocess.java_parser.tokenizers.ContinueStatement,
    preprocess.java_parser.tokenizers.ReturnStatement,
    preprocess.java_parser.tokenizers.ThrowStatement,
    preprocess.java_parser.tokenizers.SynchronizedStatement,
    preprocess.java_parser.tokenizers.TryStatement,
    preprocess.java_parser.tokenizers.SwitchStatement,
    preprocess.java_parser.tokenizers.BlockStatement,
    preprocess.java_parser.tokenizers.SwitchStatementCase,
    preprocess.java_parser.tokenizers.TryResource,
    preprocess.java_parser.tokenizers.CatchClause,
    preprocess.java_parser.tokenizers.CatchClauseParameter,
    preprocess.java_parser.tokenizers.ForControl,
    preprocess.java_parser.tokenizers.EnhancedForControl,
    preprocess.java_parser.tokenizers.BasicType,
    preprocess.java_parser.tokenizers.MemberReference,
    preprocess.java_parser.tokenizers.ReferenceType
]


def main():
    # Parse arguments.
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-train-model", action="store_true", default=False)
    arg_parser.add_argument("-eval-explanation", action="store_true", default=False)
    arg_parser.add_argument("-preprocess-dataset", action="store_true", default=False)
    arg_parser.add_argument("--project", type=str, help="The project to be evaluated. Can be one of xalan, lucene and poi.")
    arg_parser.add_argument("--output", type=str, help="The output directory.")
    arg_parser.add_argument("--model", type=str, help="The used defect prediction model. Can be one of TF, TFIDF, DBN, CNN and LSTM.")
    arg_parser.add_argument("--k", type=int, help="The features used for explanation.")
    arg_parser.add_argument("--explanators", type=str, help="The used explanator. Can be one of RG, LIME and WO (Random Guessing, LIME, and Word Omission respectively).")
    args = arg_parser.parse_args()

    project = args.project
    if project not in project_mapper.keys():
        raise ValueError("Argument '--project' must be one of lucene, poi, and xalan, got %s." % project)

    dataset = project_mapper[project]
    output_filename = args.output
    num_features_captured = args.k if args.k is not None else 5

    action = "Unknown"
    if args.train_model:
        action = "model_training"
    elif args.preprocess_dataset:
        action = "dataset_processing"
    elif args.eval_explanation:
        action = "explanation_evaluating"

    model_used = args.model
    if args.preprocess_dataset is False and model_used is None:
        raise ValueError("Argument '--model' must be specified.")

    # Initialize the output.
    output_prefix = "%s.%s.%s" % (output_filename, action, dataset)
    if action == "model_training":
        output_prefix += ("." + model_used)
    elif action == "explanation_evaluating":
        output_prefix += ("." + model_used + "." + str(num_features_captured))

    Out.init(data=output_prefix + ".result.txt",
             logging=output_prefix + ".txt")

    source_project, target_project = get_project_name(dataset)
    if args.preprocess_dataset is True:
        Out.write("Process dataset %s from source files." % dataset)
        ds = preprocess.dataset.Dataset(data_dir="./test/data/promise",
                                        source_project=source_project,
                                        target_project=target_project,
                                        ast2seq=AST2Seq(accepted_tokens=nodes))
        preprocess.dataset.Dataset.save(ds, "./test/data/pickled/promise-" + dataset + ".data")
        pass
    else:
        # Loading dataset
        Out.write_time()
        Out.write("Load dataset " + dataset + " from pickled file.")
        # Load dataset from pickled file.
        ds = preprocess.dataset.Dataset.load("./test/data/pickled/promise-" + dataset + ".data")
        # Build global dictionary for code tokens.
        global_dictionary = GlobalDictionary(projects=[ds.source_project, ds.target_project])
        # Get training data and test data.
        source_classes, source_seq, source_static_metrics, source_labels, source_files = ds.source_project.data(show_token_prefix=False)
        y = source_labels
        X = source_seq
        target_classes, target_seq, target_static_metrics, target_labels, target_files = ds.target_project.data(show_token_prefix=False)
        XX = target_seq
        yy = target_labels
        # Here we define the evaluated defect prediction models.
        # TODO More models can be added here just like the above models with a function and assignment to the available_models.
        available_models = dict()
        common_settings = {"global_dictionary": global_dictionary,
                           "resampler": OversamplingResampler(),
                           "handle_imbalance": True,
                           "checkpoint": "./trained_checkpoints/" + dataset}
        load_checkpoint_settings = {"load_checkpoint": True}

        def cnn_model(X, y, retrain=False):
            cnn_settings = {"filter_num": 10, "hidden_num": 100, "filter_length": 5, "embedding_size": 30}
            if retrain:
                CNNModel(**merge_dict(common_settings, cnn_settings)).fit(X, y).save()
            return CNNModel(**merge_dict(common_settings, cnn_settings, load_checkpoint_settings))

        available_models["CNN"] = cnn_model

        def dbn_model(X, y, retrain=False):
            dbn_settings = {"hidden_layers": 4, "nodes": 64, "iter": 200, "classifier": RandomForestClassifier()}
            if retrain:
                DBNModel(**merge_dict(common_settings, dbn_settings)).fit(X, y).save()
            return DBNModel(**merge_dict(common_settings, dbn_settings, load_checkpoint_settings))

        available_models["DBN"] = dbn_model

        def lstm_model(X, y, retrain=False):
            lstm_settings = {"lstm_units": 128, "batch_size": 28, "embedding_size": 30, "dropout": 0.05, "epochs": 1}
            if retrain:
                LSTMModel(**merge_dict(common_settings, lstm_settings)).fit(X, y).save()
            return LSTMModel(**merge_dict(common_settings, lstm_settings, load_checkpoint_settings))

        available_models["LSTM"] = lstm_model

        def tf_model(X, y, retrain=False):
            tf_settings = {"normalization": True}
            if retrain:
                TFIDFModel(**merge_dict(common_settings, tf_settings)).fit(X, y).save()
            return TFIDFModel(**merge_dict(common_settings, tf_settings, load_checkpoint_settings))

        available_models["TF"] = tf_model

        def tfidf_model(X, y, retrain=False):
            tfidf_settings = {"normalization": True}
            if retrain:
                TFIDFModel(**merge_dict(common_settings, tfidf_settings)).fit(X, y).save()
            return TFIDFModel(**merge_dict(common_settings, tfidf_settings, load_checkpoint_settings))

        available_models["TFIDF"] = tfidf_model

        if model_used not in available_models.keys():
            raise ValueError("No such model available: %s" % model_used)

        if args.train_model is True:
            # Training the source code-based models.
            Out.write_time()
            print("Training model: ", model_used)
            model_name = model_used
            model_object = available_models[model_name](X, y, True)
            Out.write_time()
            Out.write("Finish training model %s. Its performance is: " % model_name)
            y_pred = dict()
            yy_pred = dict()
            y_pred[model_name] = model_object.predict(X)
            yy_pred[model_name] = model_object.predict(XX)
            table = PrettyTable()
            table.field_names = ["", "Training Project (Source Project)", "Test Project (Target Project)"]
            table.add_row(["AUC", roc_auc_score(y, y_pred[model_name]), roc_auc_score(yy, yy_pred[model_name])])
            table.add_row(["F1", f1_score(y, y_pred[model_name]), f1_score(yy, yy_pred[model_name])])
            table.add_row(["Recall", recall_score(y, y_pred[model_name]), recall_score(yy, yy_pred[model_name])])
            table.add_row(["Precision", precision_score(y, y_pred[model_name]), precision_score(yy, yy_pred[model_name])])
            Out.write(str(table))
        elif args.eval_explanation is True:
            # Evaluating the local explanation methods.
            Out.write_time()
            Out.write("Model: %s " % model_used)
            Out.write("Number of features used for explanation: %d" % num_features_captured)
            Out.write("Used explanator(s): %s" % args.explanators)
            used_explanators = args.explanators.split(",")
            # First load model from checkpoint.
            model_name = model_used
            model_object = available_models[model_name](X, y, False)
            y_pred = {}
            yy_pred = {}
            y_pred[model_name] = model_object.predict(X)
            yy_pred[model_name] = model_object.predict(XX)
            Out.write("Used model %s, which performance is" % model_name)
            table = PrettyTable()
            table.field_names = ["", "Training Project (Source Project)", "Test Project (Target Project)"]
            table.add_row(["AUC", roc_auc_score(y, y_pred[model_name]), roc_auc_score(yy, yy_pred[model_name])])
            table.add_row(["F1", f1_score(y, y_pred[model_name]), f1_score(yy, yy_pred[model_name])])
            table.add_row(["Recall", recall_score(y, y_pred[model_name]), recall_score(yy, yy_pred[model_name])])
            table.add_row(["Precision", precision_score(y, y_pred[model_name]), precision_score(yy, yy_pred[model_name])])
            Out.write(str(table))
            # Then evaluate the local explanation methods.
            Out.write("Evaluating local explanation methods.")
            Out.write_data("Explanator,Module Name,Actual Label,Predicted Label,AOPC,FDF,EP")
            # Below are 2 variables to summarize the metrics for directly output.
            # The number of explained instance.
            explained_instance_num = 0
            # The sum of each individual metric.
            results_sum = dict()
            for explanator in used_explanators:
                results_sum[explanator] = {"aopc": 0, "pdf": 0, "ep": 0}
            for class_id in range(0, len(yy)):
                class_for_explanation = target_classes[class_id]
                Out.write("%d / %d, %s, Actual %d, Predicted %d" % (class_id,
                                                                    len(yy),
                                                                    class_for_explanation,
                                                                    yy[class_id],
                                                                    yy_pred[model_name][class_id]))
                # We assume the clean instances are trivial.
                if yy_pred[model_name][class_id] == 0:
                    continue
                # Load java source for explanation.
                java_file = JavaFile(class_name=class_for_explanation,
                                     ast=ds.target_project.abstract_syntax_tree[class_for_explanation])
                token_count, token_sequence = AST2Seq(accepted_tokens=nodes).parse(java_file)
                # Build token-level feature(s).
                token_level_instance = build_atomic_features(token_sequence, lambda x: x)

                def count_unique_features(instance):
                    unique_features = set()
                    for feature in instance.features():
                        unique_features.add(feature)
                    return len(unique_features)

                def eval_explanation(explanator, instances, score_threshold=0.0, golden_label=1):
                    """
                    Function for evaluate a explanator with given instance.
                    """

                    def pretty_print_key_features(explanation):
                        Out.write("Key features for this prediction are:")
                        for elem in explanation:
                            for feat in elem.key_features:
                                str_list = []
                                for af in feat[0]:
                                    str_list.append(str(af))
                                feat_str = ", ".join(str_list)
                                Out.write("  " + str(feat[1]) + ", " + feat_str)

                    # Explain.
                    expr = explanator.explain(instances)

                    # Output key features.
                    pretty_print_key_features(expr)

                    # Calculate metrics.
                    dict_score = deletion_metrics(instances, expr, explanator.classifier, metrics=["aopc", "fdf"],
                                                  score_threshold=score_threshold, golden_label=golden_label)
                    ep_score = explanation_precision(instances, expr, explanator.classifier, score_threshold=score_threshold,
                                                     golden_label=golden_label)

                    # Write to the report file.
                    Out.write_data(explanator.explanator_name()
                                   + "," + class_for_explanation
                                   + "," + str(yy[class_id])
                                   + "," + str(yy_pred[model_name][class_id])
                                   + "," + str("%.6f" % dict_score['aopc'])
                                   + "," + str(dict_score['fdf'])
                                   + "," + str(ep_score))

                    # Write to the logging file.
                    Out.write("AOPC: " + str("%.6f" % dict_score['aopc'])
                              + ", PDF: " + str(dict_score['fdf'])
                              + ", EP: " + str(ep_score))

                    return {"aopc": dict_score["aopc"], "pdf": dict_score["fdf"], "ep": ep_score}

                token_level_feature_num = count_unique_features(token_level_instance)
                Out.write("    Unique features: " + str(token_level_feature_num))

                explanators = dict()
                # Use 5000 perturbed local samples for LIME explanation.
                explanators["LIME"] = LIME(classifier=model_object,
                                           num_samples=5000,
                                           num_features=min(num_features_captured, token_level_feature_num))
                # The score function for WordOmission is the probability of given sequence.
                explanators["WO"] = TokenOmission(classifier=model_object,
                                                  scoring_func=lambda seq_of_atomic: model_object.predict_proba([features_to_tokens(seq_of_atomic)])[0][1],
                                                  num_features=min(num_features_captured, token_level_feature_num))
                explanators["RG"] = RandomSelection(classifier=model_object,
                                                    num_features=min(num_features_captured, token_level_feature_num))

                # TODO More novel explanation methods can be added here for evaluation.

                for explanator in used_explanators:
                    if explanator not in explanators.keys():
                        raise ValueError("Unknown explanator %s." % explanator)

                # Evaluate the explanator and record results.
                for explanator in used_explanators:
                    Out.write(explanator)
                    result = eval_explanation(explanators[explanator], [token_level_instance], golden_label=yy_pred[model_name][class_id])
                    for metric in result.keys():
                        results_sum[explanator][metric] += result[metric]
                    Out.write("")

                Out.flush()
                explained_instance_num += 1

            # Summarize the automatic metrics of evaluated local explanation methods.
            pt = PrettyTable()
            pt.field_names = ["Local Explanation Method", "AOPC", "PDF", "EP"]
            for explanator in used_explanators:
                aopc = results_sum[explanator]["aopc"] / explained_instance_num
                pdf = results_sum[explanator]["pdf"] / explained_instance_num
                ep = results_sum[explanator]["ep"] / explained_instance_num
                pt.add_row([explanator, aopc, pdf, ep])
            Out.write(str(pt))
        else:
            print("Wrong operation argument.")
            return
    Out.write("Done.")
    Out.close()
    return


if __name__ == "__main__":
    main()
