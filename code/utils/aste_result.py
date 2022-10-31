import os
import time
from . import append_json, save_json, mkdir_if_not_exist

polarity_map = {
    'NEG': 0,
    'NEU': 1,
    'POS': 2
}
polarity_map_reversed = {
    0: 'NEG',
    1: 'NEU',
    2: 'POS'
}

class F1_Measure:
    def __init__(self):
        self.pred_set = set()
        self.true_set = set()

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_set.add((idx, tuple(pred)))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_set.add((idx, tuple(true)))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_set, self.true_set)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_set, true_set):
        intersection = pred_set.intersection(true_set)
        _p = len(intersection) / len(pred_set) if pred_set else 1
        _r = len(intersection) / (len(true_set)) if true_set else 1
        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0
        return f1, _p, _r



class NER_F1_Measure(F1_Measure):
    def __init__(self, entity_types):
        super().__init__()
        self.entity_types = entity_types
        
    def report(self):
        for entity_type in self.entity_types:
            name = '_'.join(entity_type)
            pred_set = self.filter(self.pred_set, entity_type)
            true_set = self.filter(self.true_set, entity_type)
            
            f1, p, r = self.cal_f1(pred_set, true_set)
            setattr(self, f'{name}_f1', f1)
            
    def filter(self, set_, entity_type):
        return_set = set()
        for type_ in entity_type:
            return_set.update(set([it for it in set_ if it[1][0] == type_]))
        return return_set


class Result:
    def __init__(self, result_json):
        self.result_json = result_json
        self.detailed_metric = None
        self.monitor = 0

    def __setitem__(self, key, value):
        self.result_json[key] = value

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, all_preds, examples):
        result_json = {}
        examples = {example['ID']: example for example in examples}

        for preds in all_preds:

            for ID in preds['ids']:
                example = examples[ID]
                pairs_true = []
                for pp in example['pairs']:
                    pl = polarity_map[pp[4]]+1
                    pairs_true.append([pp[0],pp[1],pp[2],pp[3],pl])
                result_json[ID] = {
                    'ID': ID,
                    'sentence': example['sentence'],
                    'pairs': pairs_true,
                    'tokens': str(example['tokens']),
                    'pair_preds': set(),
                }
            for (i, a_start, a_end, b_start, b_end, pol) in preds['pair_preds']:
                ID = preds['ids'][i]
                result_json[ID]['pair_preds'].add((a_start, a_end, b_start, b_end, pol))  
        return cls(result_json)


    def cal_metric(self):
        b_pair_f1 = F1_Measure()
        for item in self.result_json.values():
            for pair_f1 in (b_pair_f1, ):
                pair_f1.true_inc(item['ID'], item['pairs'])

            b_pair_f1.pred_inc(item['ID'], item['pair_preds'])

        b_pair_f1.report()

        detailed_metrics = {
            'pair_f1': b_pair_f1['f1'],
            'pair_p':b_pair_f1['p'],
            'pair_r':b_pair_f1['r'],
            }

        self.detailed_metrics = detailed_metrics
        self.monitor = b_pair_f1['f1']

    def report(self):
        print(f'monitor: {self.monitor:.4f}', end=" ** ")
        for metric_names in (('pair_f1','pair_p','pair_r'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()

    def save(self, dir_name, args):
        mkdir_if_not_exist(dir_name)
        current_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
        current_day  = time.strftime("%Y-%m-%d", time.localtime())
        
        result_file_name = os.path.join(dir_name, f'val_results_{self.monitor*10000:4.0f}_{current_time}.txt')
        performance_dir = os.path.join(os.path.dirname(os.path.dirname(dir_name)), 'performance')

        performance_dir = os.path.join(performance_dir, current_day)
        performance_file_name = os.path.join(performance_dir, f'{args.cuda_ids}.txt')

        for key, item in self.result_json.items():
            item['pair_preds'] = list(item['pair_preds'])

        save_json(list(self.result_json.values()), result_file_name)
        print('## save result to', result_file_name)

        description = f'{args.data_dir}, lr={args.learning_rate}, seed={args.seed}, model_name_or_path={args.model_name_or_path}'
        detailed_metrics = {k: (v if type(v) in (int, float) else v.item()) for k,v in self.detailed_metrics.items()}

        append_json(performance_file_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        append_json(performance_file_name, f'{description} {self.monitor*10000:4.0f}')
        append_json(performance_file_name, f'{args.span_pruning} {args.seq2mat} {args.num_d} {args.table_encoder} {args.num_table_layers}')
        append_json(performance_file_name, detailed_metrics)
        append_json(performance_file_name, '')


