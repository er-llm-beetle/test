### İstədiyiniz modeli local-da ollama istifadə edərək evaluate etmək üçün aşağıdakı addımları izləyə bilərsiniz:


##### 1. Modelin Konfiqurasiyası
.yaml faylını tənzimləyin və istədiyiniz modelin adını yazın. Əgər model default olaraq Ollama-da mövcud deyilsə, onu modelfile və .gguf ilə konfiqurasiya etməlisiniz.

##### 2. env Faylının Yaradılması
.env faylı yaradın: Bu faylda lazım olan key-i müəyyən edin. Misal üçün, nvidia llama3.1 405b key istifadə edilə bilər. key-i necə adlandırmaq lazım olduğunu öyrənmək üçün base.py-də os.getenv() içindəki mətnə baxın.

##### 3. Virtual Mühitin Yaradılması
venv faylı yaradın, daha sonra onu aktivləşdirin və lazimi kitabxanaları yükləyin. 
Bunun üçün terminalda aşağıdakı əmri icra edin:

Windows:
```bash
mkdir my_project
cd my_project
python -m venv venv
```

MacOS:
```bash
mkdir my_project
cd my_project
python3 -m venv venv
```


Daha sonra virtual environment-i aktivləşdirmək üçün:

Windows:
```bash
venv\Scripts\activate
```

MacOS:
```bash
source venv/bin/activate
```


requirements.txt faylını quraşdırmaq üçün isə aşağıdakı əmri yerinə yetirin:

```bash
pip install -r requirements.txt
```


##### 4. Modulları Yoxlamaq
Modul və kitabxanaların uyğun və tam olduğuna və yükləndiyinə əmin olun.

##### 5. Modeli Run Etmək
Modeli evaluate etmək üçün bu faylı icra edin:
```bash
python evaluate_yaml.py
```

Alternativ olaraq: Hissə-hissə icra etmək üçün evaluate_yaml_chunked_main.py faylını istifadə edə bilərsiniz:
```bash
python evaluate_yaml_chunked_main.py
```


Bu addımlardan sonra modeliniz lokalda işləməyə hazır olacaq. Hər hansı sualınız yaranarsa, soruşmaqdan və mənimlə əlaqə yaratmaqdan çəkinməyin.
