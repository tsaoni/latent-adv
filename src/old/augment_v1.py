from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from lib import *

def data_augment_by_mask_prediction(
        src_filename, tgt_filename, aug_src_filename, aug_tgt_filename, 
        mask_ratio=0.3, 
        augment_num=5, 
        generate_kwargs=None, # num_beams, max_length_inc, min_length_inc, 
    ):
    model_name = 'facebook/bart-large'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    config = BartConfig.from_pretrained(model_name, output_hidden_states=True)
    model = BartForConditionalGeneration.from_pretrained(model_name, config=config)

    with open(src_filename, 'r') as src_f, open(tgt_filename, 'r') as tgt_f, \
            open(aug_src_filename, 'w') as aug_src_f, open(aug_tgt_filename, 'w') as aug_tgt_f:
        first_write = True
        while True:
            text, label = src_f.readline().strip(), tgt_f.readline().strip()
            text_length = len(text.split())
            if not text: break
            def random_mask(text, mask_ratio, argument_num):
                mask_token = tokenizer.mask_token
                mask_word_num = int(len(text.split()) * mask_ratio)
                masked_text_list = []
                while len(masked_text_list) < argument_num:
                    masked_text = copy.deepcopy(text)
                    masked_word_list = masked_text.split()
                    masked_num = 0
                    while masked_num < mask_word_num:
                        # determine the position to mask
                        unmask_pos = [i for i in range(len(masked_word_list) - 1) if not masked_word_list[i] == mask_token]
                        if len(unmask_pos) == 0: return [mask_token, masked_word_list[-1]]
                        else:
                            import random
                            idx = random.randint(0, len(unmask_pos) - 1)
                            masked_word_list[unmask_pos[idx]] = mask_token
                        masked_num += 1
                    
                    filter_cond = lambda idx, word_list: idx > 0 and word_list[idx - 1] == word_list[idx] and word_list[idx] == mask_token
                    masked_word_list = [word for i, word in enumerate(masked_word_list) if not filter_cond(i, masked_word_list)]
                    masked_text_list.append(" ".join(masked_word_list))

                return masked_text_list

            masked_text_list = random_mask(text, mask_ratio, augment_num)
            text_id_length = len(tokenizer.encode(text))
            batch = tokenizer.prepare_seq2seq_batch(
                masked_text_list, 
                max_length=text_id_length, 
                return_tensors='pt', 
            )
            output_sequences = model.generate(
                batch['input_ids'], 
                attention_mask=batch["attention_mask"],
                num_beams=generate_kwargs['num_beams'], 
                min_length=text_id_length + generate_kwargs['min_length_inc'], 
                max_length=text_id_length + generate_kwargs['max_length_inc'], 
                # early_stopping=True, 
            )
            augmented_text_list = tokenizer.batch_decode(output_sequences.tolist(), skip_special_tokens=True)
            output = list(map(lambda x: x.strip(), augmented_text_list))
            output.append(text) # remember to add the original text

            prefix = '' if first_write else '\n'
            first_write = False 
            output_str = prefix + '\n'.join(output)
            aug_src_f.write(output_str)
            labels = [label] * len(output)
            label_str = prefix + '\n'.join(labels)
            aug_tgt_f.write(label_str)

            # print('mask text: \n', '\n'.join(masked_text_list))
            # print('filled text: \n', '\n'.join(argumented_text_list))


def text_postprocessing(src_file, tgt_file):
    import re
    with open(src_file, 'r') as src_f, open(tgt_file, 'w') as tgt_f:
        first_write = True
        while True:
            text = src_f.readline().strip()
            if not text: return 
            # text = re.sub(r'\[.*?\]', '', text) # remove text between []
            text = text.replace('[', '').replace(']', '') # remove []
            text = re.sub(r'^[.,\s]+', '', text) # remove space, comma and period in the beginning of sentence
            text = re.sub(r'\,+', ',', text) # remove ,,,
            # remove parentheses
            matches = re.findall(r'\((.*?)\)', text)
            if len(matches) > 0 and len(matches[0].split()) > 1:
                text = text.replace('(', '').replace(')', '')
            else:
                text = re.sub(r'\(.*?\)', '', text)

            # remove ......
            text_list = np.array([s.strip() for s in text.split('.')])
            text_list = text_list[text_list != '']
            text = '. '.join(text_list)
            # remove website. 
            text = text.split('https://')[0]
            text = text.split('http://')[0]
            # remove number in the front
            text = re.sub(r'^\d+\.\s+', '', text)
            # remove [a] [b] [c] [d]
            text = text.replace('[a] [b] [c] [d]', '')

            text = ' '.join(text.split())
            prefix = '' if first_write else '\n'
            first_write = False
            tgt_f.write(prefix + text)

            
if __name__ == '__main__':
    do_augment = False
    do_postprocess = True

    src_file = '../data/rotten_tomatoes/cls/train.source'
    tgt_file = '../data/rotten_tomatoes/cls/train.target'
    argument_src_file = '../data/rotten_tomatoes/cls/augmented/train.source'
    argument_tgt_file = '../data/rotten_tomatoes/cls/augmented/train.target'
    post_src_file = '../data/rotten_tomatoes/cls/post-augmented/train.source'
    post_tgt_file = '../data/rotten_tomatoes/cls/post-augmented/train.target'
    os.makedirs('../data/rotten_tomatoes/cls/augmented', exist_ok=True)
    os.makedirs('../data/rotten_tomatoes/cls/post-augmented', exist_ok=True)
    generate_kwargs = dict(
        num_beams=5, 
        max_length_inc=20, 
        min_length_inc=2, 
    )

    if do_augment:
        data_augment_by_mask_prediction(
            src_file, tgt_file, argument_src_file, argument_tgt_file, 
            generate_kwargs=generate_kwargs
        )
    if do_postprocess:
        os.system(f'cp {argument_tgt_file} {post_tgt_file}')
        text_postprocessing(argument_src_file, post_src_file)
