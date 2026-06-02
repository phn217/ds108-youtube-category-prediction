import pandas as pd
from groq import Groq
import time
import re

# 1. CẤU HÌNH API GROQ
API_KEY = "YOUR_GROQ_API_KEY_HERE"  # Thay bằng API Key của bạn
client = Groq(api_key=API_KEY)

def generate_tags_from_title(title):
    stopwords = {'the','and','for','with','this','that','are','was','but','not','you','all','from','have','they','has','will','how'}
    words = re.findall(r"[A-Za-z0-9']{3,}", str(title))
    tags = [w.lower() for w in words if w.lower() not in stopwords]
    return '|'.join(tags[:10]) if tags else ''

# 2. ĐỌC DỮ LIỆU
print("Đang đọc file ds108_raw.csv...")
df = pd.read_csv('ds108_raw.csv', encoding='utf-8')

if 'tags_source' not in df.columns:
    df['tags_source'] = 'original'

# Tách riêng 900 dòng cho AI và phần còn lại cho NLP
missing_indices = df[df['tags'].isnull()].index
AI_SAMPLE_SIZE = 900
ai_indices = missing_indices[:AI_SAMPLE_SIZE]
nlp_indices = missing_indices[AI_SAMPLE_SIZE:]

api_blocked = False 

# 3. CHẠY AI (Llama 3.1) CHO 900 DÒNG (Dự kiến ~31 phút)
print(f"Bắt đầu xử lý {AI_SAMPLE_SIZE} dòng bằng Llama 3.1 (Groq)...")
count = 0

for idx in ai_indices:
    title = df.loc[idx, 'title']
    category = df.loc[idx, 'category_name']
    
    if not api_blocked:
        try:
            prompt = f'Tạo 5 tags (cách nhau bằng |) cho video "{title}". Không giải thích.'
            
            # ĐÃ ĐƯỢC CẬP NHẬT: Sử dụng model llama-3.1-8b-instant đang hoạt động ổn định
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  
                temperature=0.3,
                max_tokens=50
            )
            
            ai_tags = chat_completion.choices[0].message.content.strip().replace('\n', '').replace('"', '').replace("'", "")
            
            df.loc[idx, 'tags'] = ai_tags
            df.loc[idx, 'tags_source'] = 'groq_llama3'
            print(f"[{count+1}/{AI_SAMPLE_SIZE}] [Llama 3.1] Đã sinh: {ai_tags[:50]}...")
            
            # Lưu Checkpoint mỗi 100 dòng
            if (count + 1) % 100 == 0:
                df.to_csv('ds108_checkpoint_hybrid.csv', index=False, encoding='utf-8')
                print(f"Đã lưu nháp an toàn tại mốc {count + 1} dòng!")
            
            # NGHỈ 2.1 GIÂY ĐỂ AN TOÀN TUYỆT ĐỐI VỚI RATE LIMIT
            time.sleep(2.1)
            
        except Exception as e:
            print(f"\nLỖI API GROQ: {e}")
            print("Kích hoạt Ngắt mạch (Circuit Breaker) -> Chuyển sang NLP!")
            api_blocked = True 
            
            df.loc[idx, 'tags'] = generate_tags_from_title(title)
            df.loc[idx, 'tags_source'] = 'nlp_fallback'
    
    else:
        df.loc[idx, 'tags'] = generate_tags_from_title(title)
        df.loc[idx, 'tags_source'] = 'nlp_fallback'
        
    count += 1

# 4. CHẠY NLP CHO PHẦN CÒN LẠI (Luồng mặc định - Chạy chưa tới 1 giây)
print(f"Bắt đầu chạy NLP cho {len(nlp_indices)} dòng cuối cùng...")
for idx in nlp_indices:
    title = df.loc[idx, 'title']
    df.loc[idx, 'tags'] = generate_tags_from_title(title)
    df.loc[idx, 'tags_source'] = 'nlp_regex'

# 5. LƯU THÀNH QUẢ
df.to_csv('ds108_raw_tags_completed.csv', index=False, encoding='utf-8')
print("Đã chạy xong Mô hình Lai!")