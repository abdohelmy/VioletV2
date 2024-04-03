import random
import wandb
from data import ImagesField, TextField, RawField,ImagesField_noncoco
from data import COCO,DataLoader,XM3600, CC3M
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer_visualgpt, VisualEncoder, ScaledDotProductAttentionMemory, ScaledDotProductAttention
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from sys import exit
import logging
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import AdamW
from torch import nn
# from accelerate import Accelerator
from datetime import datetime
from data.dataset import Dataset
import pandas as pd
from torch.nn import DataParallel as DDP
from models.captioning_model import CaptioningModel
from PIL import Image
import glob

# class XM3600(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir, field):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.cross_modal = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.field = field
#         self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
#         # imgs_path = os.path.join(self.root_dir,
#         #                         self.cross_modal.iloc[:, 0]+".jpg")
#         # image_files = glob.glob(imgs_path)
#         # # img_name = os.path.join(self.root_dir,
#         # #                         self.cross_modal.iloc[:, 0]+".jpg")
#         # img_list
#         # image = Image.open(img_name)
#         # image = np.array(image)
#         # inputs = self.processor(images=image, return_tensors="pt")
#         # img = inputs["pixel_values"].squeeze(0)
#         caption = self.cross_modal.iloc[:, 1]
#         caption = list(caption)
#         tokenized = PTBTokenizer.tokenize(caption)
#         self.caps = self.field.process(tokenized)
#     def __len__(self):
#         return len(self.cross_modal)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
# #field.process
# #PTBTokenizer.tokenize
# #clip preprocess

#         img_name = os.path.join(self.root_dir,
#                                 self.cross_modal.iloc[idx, 0]+".jpg")
#         image = Image.open(img_name)
#         image = np.array(image)
#         inputs = self.processor(images=image, return_tensors="pt")
#         img = inputs["pixel_values"].squeeze(0)
#         # caption = self.cross_modal.iloc[idx, 1]
#         # caption = np.array([caption])
#         # tokenized = PTBTokenizer.tokenize(caption)
#         # caps = self.field.process(tokenized)
#         caps = self.caps[idx]
#         sample = {'image': img, 'caption': caps}

#         # if self.transform:
#         #     sample = self.transform(sample)

#         return sample
    
def check_memory(cuda_device):
    """ Check the total memory and occupied memory for GPU """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_memory(cuda_device):
    """ Create a large tensor and delete it.
    This operation occupies the GPU memory, so other processes cannot use the occupied memory.
    It is used to ensure that this process won't be stopped when it requires additional GPU memory.
    Be careful with this operation. It will influence other people when you are sharing GPUs with others.
    """
    for i,gpu in enumerate(cuda_device.split(',')):
        total, used = check_memory(gpu)
        cuda = torch.device('cuda:'+str(i))
        total = int(total)
        used = int(used)
        max_mem = int(total * 0.90)
        print('Total memory: ' + str(total) + ', used memory: ' + str(used))
        block_mem = max_mem - used
        if block_mem > 0:
            x = torch.FloatTensor(256, 1024, block_mem).to(device=cuda)
            del x



def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):


                detections, captions = detections.to(device), captions.to(device)
                out,past = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, 63999), captions.view(-1)) #vocab size
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, exp_name=None, epoch=0):
    import itertools
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.module.beam_search(images, 20, text_field.vocab.stoi['<|endoftext|>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
                #gts['%d_%d' % (it, i)] = text_field.decode(gts_i[1::], join_words=False)
                #plt.imshow(np.transpose(images[i].cpu().detach().numpy(), (1, 2, 0)))
                #text_field.decode(gts_i[1::], join_words=False)

            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)



    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, text_field,gpt_optimizer,dataloader_eval,args):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    # accelerator = Accelerator()
    # model, gpt_optimizer, dataloader = accelerator.prepare(
    #  model, gpt_optimizer, dataloader)
    model = DDP(model.module)
    model.to(device)
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            # print(detections["pixel_values"].shape)
            # detections = detections["pixel_values"].squeeze(1)

            detections, captions = detections.to(device), captions.to(device)


            out,past= model(detections, captions)

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, 63999), captions_gt.view(-1)) #vocab size

            loss.backward()

            # accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)

            gpt_optimizer.step()
            gpt_optimizer.zero_grad()


            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, cider, text_field,gpt_optimizer,args):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    model = DDP(model.module)
    running_loss = .0
    seq_len = 40
    beam_size = 5
    
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.module.beam_search(detections, seq_len, text_field.vocab.stoi['<|endoftext|>'],
                                                beam_size, out_size=beam_size)

            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))

            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()

            loss.backward()

            if (it + 1) % args.gradient_accumulation_steps == 0 or (it+1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_grad_norm)
                gpt_optimizer.step()
                gpt_optimizer.zero_grad()


            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()



    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    tokenizer_pool.close()
    return loss, reward, reward_baseline



if __name__ == '__main__':
    now = datetime.now()

    current_time = now.strftime("%d-%b-%H:%M:%S")
    parser = argparse.ArgumentParser(description='VisualGPT')
    parser.add_argument('--exp_name', type=str, default='visualGPT'+str(current_time))
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--head', type=int, default=12)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--random_seed', type = int, default="42")
    parser.add_argument('--gpt_model_type',type=str, default= "gpt")
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--log_file',type = str, default="log/visualGPT.txt")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")


    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--optimizer_type', type= str, default = "adamw")
    parser.add_argument('--max_grad_norm', default=1.0, type = float)
    parser.add_argument('--train_percentage', default=1.0, type = float)
    parser.add_argument('--split_train_data', action="store_true")
    parser.add_argument('--reinforcement_lr',type = float, default=1e-5)
    parser.add_argument("--decoder_layer", type= int, default = 12)
    parser.add_argument("--encoder_layer",type=int, default=3)
    parser.add_argument("--tau",type=float, default = 0.0)

    args = parser.parse_args()


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    

    os.environ["WANDB_API_KEY"] = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    occupy_memory(os.environ["CUDA_VISIBLE_DEVICES"])
    n_gpus = torch.cuda.device_count()

    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(args)
    #
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    config = dict(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_percentage = args.train_percentage,
        name = args.exp_name
    )
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    # image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)
   # image_field2 = ImageDetectionsField(detections_path="./flicker.hdf5 ", max_detections=50, load_in_tmp=False)
    image_field = ImagesField(images_path=args.features_path, load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<|endoftext|>', eos_token='<|endoftext|>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder,train_percentage=args.train_percentage,split_train_data=args.split_train_data)
    train_dataset, val_dataset, test_dataset = dataset.splits
    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_GPT_vocab("jasminemodel/eyad-bs/vocab.json")
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))



    # Model and dataloaders
    encoder = VisualEncoder(args.encoder_layer, 0, attention_module=ScaledDotProductAttention)
    model = Transformer_visualgpt(text_field.vocab.stoi['<|endoftext|>'], encoder, args.gpt_model_type, args.decoder_layer,tau=args.tau)
    model = DDP(model)
    model.to(device)
    data = torch.load('./saved_models/jasmine_clip_adapter_best.pth')
    torch.set_rng_state(data['torch_rng_state'])
    torch.cuda.set_rng_state(data['cuda_rng_state'])
    np.random.set_state(data['numpy_rng_state'])
    random.setstate(data['random_rng_state'])
    model.module.load_state_dict(data['state_dict'],strict=False)

    
    #dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})




    # dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    # dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})




    # total_step_number = int(len(train_dataset)/(args.batch_size * args.gradient_accumulation_steps)*100)
 

    # if args.optimizer_type =="adamw":
        
    #     gpt_optimizer = AdamW(model.module.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-8)
  
    # elif args.optimizer_type =="adam":
    #     optimizer = Adam(model.module.parameters(), lr = args.lr)

 


    # loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<|padding|>'])
    # use_rl = False
    # best_cider = .0
    # best_loss = np.inf
    # patience = 0
    # start_epoch = 0

    # use_rl=True

        #dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        #                            drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,drop_last=True)
    #dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
        #                            num_workers=args.workers)
    #dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    #dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

###############################################################
    ref_caps_train = list(train_dataset.text)
    cider_train=ref_caps_train
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    image_field_XM = ImagesField_noncoco()
# Pipeline for text
    text_field_XM = TextField(init_token='<|endoftext|>', eos_token='<|endoftext|>', lower=True, tokenize='spacy',remove_punctuation=True, nopoints=False)
    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
            print("Building vocabulary")
            text_field.build_GPT_vocab("jasminemodel/eyad-bs/vocab.json")
    else:
            text_field_XM.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))
    dataset_val = XM3600(image_field_XM, text_field_XM, ann_root = "./annotations/cross_3600_fixed_order.csv", img_root = "./cross-modal")

    data = torch.load('./saved_models/jasmine_clip_more_best.pth')
    torch.set_rng_state(data['torch_rng_state'])
    torch.cuda.set_rng_state(data['cuda_rng_state'])
    np.random.set_state(data['numpy_rng_state'])
    random.setstate(data['random_rng_state'])
    model.module.load_state_dict(data['state_dict'],strict=False)
    #dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False, num_workers=args.workers, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    #scores = evaluate_metrics(model, dataloader_val,text_field_XM )
    scores = evaluate_metrics(model, dataloader_val,text_field )
    val_cider = scores['CIDEr']
    print("Cider score so far  "+str(scores['CIDEr']))
    #################################################################
