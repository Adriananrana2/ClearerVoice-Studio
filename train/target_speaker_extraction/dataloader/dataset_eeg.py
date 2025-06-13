import numpy as np
import math, os, csv

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf

from .utils import DistributedSampler

def get_dataloader_eeg(args, partition): #(YamlConfig, 'T/V/T') [FIXME]
    datasets = dataset_eeg(args, partition)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler,
            collate_fn=custom_collate_fn)
    
    return sampler, generator

def custom_collate_fn(batch):
    a_mix, a_tgt, ref_tgt = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt) 
    ref_tgt = torch.tensor(ref_tgt) 
    return a_mix, a_tgt, ref_tgt

class dataset_eeg(data.Dataset):
    def __init__(self, args, partition): #(YamlConfig, 'T/V/T') [FIXME]
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.ref_sr=args.ref_sr
        self.speaker_no=args.speaker_no
        self.batch_size=args.batch_size

        self.mix_lst_path = args.mix_lst_path
        self.audio_direc = args.audio_direc
        self.stimulus_direc = args.stimulus_direc
        self.eeg_direc = args.reference_direc
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        file_names = [x.split(',')[1] for x in mix_lst]
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

        self.eeg_dict={}
        for item in file_names:
            filename = item + "response.npy"
            eeg_path = os.path.join(self.eeg_direc, filename)
            eeg_data = np.load(eeg_path)
            self.eeg_dict[eeg_path] = eeg_data


    def __getitem__(self, index):
        mix_audios=[]
        tgt_audios=[]
        tgt_eegs=[]
        
        batch_lst = self.minibatch[index]
        min_length_second = 19.0      # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second*self.ref_sr)
        min_length_audio = math.floor(min_length_second*self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length*self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length*self.audio_sr)

        for line_cache in batch_lst:
            subject_trial = line_cache.split(',')[1]

            # load target eeg
            filename = os.path.join(self.eeg_direc, subject_trial + "response.npy")
            eeg_data = self.eeg_dict[filename]
            eeg_start = 0
            eeg_end = eeg_start + min_length_eeg
            eeg_tgt = eeg_data[eeg_start:eeg_end,:]

            # load tgt audio
            tgt_audio_path = self.audio_direc + subject_trial + "soli.wav"
            start = 0
            end = start + min_length_audio
            a_tgt, _ = sf.read(tgt_audio_path, start=int(start), stop=int(end), dtype='float32')

            # load stimulus audio
            stimulus_audio_path = self.stimulus_direc + subject_trial + "stimulus.wav"
            start = 0
            end = start + min_length_audio
            a_mix, _ = sf.read(stimulus_audio_path, start=int(start), stop=int(end), dtype='float32')

            # audio normalization
            max_val = np.max(np.abs(a_mix))
            if max_val > 1:
                a_mix /= max_val
                a_tgt /= max_val

            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_eegs.append(eeg_tgt)

        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), np.asarray(tgt_eegs, dtype=np.float32)


    def __len__(self):
        return len(self.minibatch)










