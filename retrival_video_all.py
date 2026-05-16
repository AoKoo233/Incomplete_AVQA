import torch
import os
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from tqdm import tqdm
import json


with open('./dataset/split_que_id/music_avqa_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)  

with open('./dataset/split_que_id/music_avqa_val.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)  

with open('./dataset/split_que_id/music_avqa_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f) 

candidata = []

for i in train_data:
        candidata.append(i["video_id"])

for i in val_data:
        candidata.append(i["video_id"])

for i in test_data:
        candidata.append(i["video_id"])

# ================================
# 配置路径
# ================================
video_frames_dir = "./datasets/Music-AVQA/avqa-frames-1fps/"  
audio_dir = "./datasets/Music-AVQA/audio/"               
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ================================
# 加载模型
# ================================
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# ================================
# 准备数据
# ================================

# 获取视频帧路径（假设文件名为 frame_001.jpg, frame_002.jpg, ...）
image_paths = sorted(
    [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir) if f.endswith(".jpg")]
)

video_paths = []
for i in candidata:
    temp = video_frames_dir + i + '/'
    if temp not in video_paths:
        video_paths.append(temp)

audio_paths = []
for i in candidata:
    temp = audio_dir + i + '.wav'
    if temp not in audio_paths:
        audio_paths.append(temp)

print(f"Found {len(audio_paths)} audio files.")

# ================================
# 加载与变换数据
# ================================
audio_embeds = []
batch_size = 8  # 根据显存可调整

for j in tqdm(audio_paths):
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data([j], device),
    }
    with torch.no_grad():
        embeds = model(inputs)[ModalityType.AUDIO]
    audio_embeds.append(embeds)
audio_embeds = torch.stack([k for k in audio_embeds], dim=0).squeeze()



vision_embeds = torch.load('/public/share/cit_ztyu/zhangjiayu/datasets/Music-AVQA/imagebind_video_(train+val).pt')

torch.save(audio_embeds, '/public/share/cit_ztyu/zhangjiayu/datasets/Music-AVQA/imagebind_audio_(train+val+test).pt')    # 保存 tensor（或任何 picklable 对象）
print(audio_embeds.shape,flush=True)


## 加载音频特征
#inputs = {
#    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
#}
#with torch.no_grad():
#    audio_embeds = model(inputs)[ModalityType.AUDIO]



similarity = torch.softmax(audio_embeds @ vision_embeds.T, dim=-1).squeeze(0)

#best_match_idx = torch.argmax(similarity).item()
#print(best_match_idx.shape)
_, topk_indices = torch.topk(similarity, k=15)
ans = {}
tot = 0
for i in audio_paths:
    ans[i.split("/")[-1].split(".")[0]] = []
    for index in topk_indices[tot]:
        ans[i.split("/")[-1].split(".")[0]].append(video_paths[index].split("/")[-2])
    tot +=1

# 指定保存JSON数据的文件路径
file_path = "(music)retrieval_form_audio_to_video_top15.json"

with open(file_path, "w") as json_file:
    json.dump(ans, json_file)


#best_match_audio = audio_paths[best_match_idx]

# ================================
# 输出结果
# ================================
#print("=== Retrieval Result ===")
#for i, (path, score) in enumerate(zip(audio_paths, similarity.tolist())):
#    print(f"{i+1:02d}. {os.path.basename(path):<30}  Similarity: {score:.4f}")

#print(f"\n>> Best matching audio: {best_match_audio}")
