# Samsung CE Challenge 2024
## How to Run
1. Pull docker image
```
docker pull yezinri/samsung:v1
```
---------------------------------------
2. Run docker
```
docker run -it -v <MODEL_DIR>:<MODEL_DIR> --runtime nvidia yezinri/samsung:v1 /bin/bash
```
ex) 
```
docker run -it -v /home/samsung/model:/home/samsung/model --runtime nvidia yezinri/samsung:v1 /bin/bash
```
---------------------------------------
3. Execute the script
```
python3 /script.py --file_path --base_path --batch_size --max_new_tokens
```
ex)
```python3 /script.py --file_path='/data.jsonl' --base_path='/home/samsung/model' --batch_size=100 --max_new_tokens=15
```
- file_path : jsonl 파일 경로 
- base_path : safetensors 경로 
- batch_size : batch size , default=100입니다. 
- max_new_tokens : generate 할 토큰 수, default=15입니다. 