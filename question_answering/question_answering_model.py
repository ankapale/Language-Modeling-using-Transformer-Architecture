import torch
from modeling import BertForQuestionAnswering
from tokenization import BertTokenizer
from squad_utils import *

def get_results(eval_features, model):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    device = "cpu"
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    return all_results


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def is_valid_start_end_index(start_index, end_index, feature):
    # We could hypothetically create invalid predictions, e.g., predict
    # that the start of the span is in the question. We throw out all
    # invalid predictions.
    if start_index >= len(feature.tokens):
        return False
    if end_index >= len(feature.tokens):
        return False
    if start_index not in feature.token_to_orig_map:
        return False
    if end_index not in feature.token_to_orig_map:
        return False
    if not feature.token_is_max_context.get(start_index, False):
        return False
    if end_index < start_index:
        return False
    length = end_index - start_index + 1
    if length > 64:
        return False
    return True

def predict_answer(example, model, tokenizer, top_n=5):
    all_features = convert_examples_to_features(
        examples=[example],
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False)
    
    all_results = get_results(all_features, model)
    
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    result_list = []

#     features = example_index_to_features[example_index]
    features = all_features

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min null score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        start_indexes = _get_best_indexes(result.start_logits, 2*top_n)
        end_indexes = _get_best_indexes(result.end_logits, 2*top_n)

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if not is_valid_start_end_index(start_index, end_index, feature):
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= top_n:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, True)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))
    return nbest

def compose_question(doc_text, question_text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    doc_tokens = []
    prev_is_whitespace = True
    for c in doc_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            
    return SquadExample(qas_id="Q1",
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=None,
                start_position=None,
                end_position=None,
                is_impossible=None)

def init_params():
    model = BertForQuestionAnswering.from_pretrained("./bert_squad/")
    tokenizer = BertTokenizer.from_pretrained("./bert_squad/", do_lower_case=True)
    device = "cpu"
    model.to(device)

def question_answering(doc_text, question_text, model, tokenizer):
    example = compose_question(doc_text, question_text)
    predictions = predict_answer(example, model, tokenizer, top_n=3)
    results = []
    for p in predictions:
        results.append((p.text, str((p.start_logit+p.end_logit)/2)))
    return results

def question_answering_terminal():
    doc_text = input('Input document String: ')
    question_text = input('Input question String: ')
    example = compose_question(doc_text, question_text)
    predictions = predict_answer(example, top_n=3)
    results = []
    for p in predictions:
        results.append((p.text, str((p.start_logit+p.end_logit)/2)))
    return results

if __name__ == "__main__":
    model = BertForQuestionAnswering.from_pretrained("./bert_squad/")
    tokenizer = BertTokenizer.from_pretrained("./bert_squad/", do_lower_case=True)
    device = "cpu"
    model.to(device)
    print(question_answering_terminal())