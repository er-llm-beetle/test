

from base import *


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model_answer_multiple_options(question, options, model, dstype) -> str:
    """
    Sends a query to the model and retrieves the response.

    Args:
        client: The Ollama client instance.
        question (str): The question to be categorized.
        options (str): The options for categorization.
        existing_answer (str): The existing answer.

    Returns:
        str: The model's response.
    """

    # client = ollama.Client()



    if dstype == 'tc':
        try:
            formatted_options = options.replace(", ", ",\n")
            options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Respond with only the letter of the chosen answer (e.g., A, B, C, etc.), without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nMənə yaxın filial tapmağa kömək edin.\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                # "content": "Question:\nMənə yaxın filial tapmağa kömək edin.\nOptions:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nAnswer:" # Reshad's updated data

            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                "content": "Question:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                # "content": "Question:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nOptions:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nAnswer:" # Reshad's updated data
            },
            {
                "role": "assistant",
                "content": "G"
            },
            {
                "role": "user",
                "content": "Question:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                # "content": "Question:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nOptions:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nAnswer:" # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                # "content": "Question:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nOptions:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nAnswer:" # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                # "content": "Question:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nOptions:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nAnswer:"    # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]

        # aze
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Sənə bir ifadə verilir və bu ifadə ilə əlaqəli müxtəlif mövzuları əks etdirən bir neçə seçim təqdim edilir. İfadənin mövzusuna uyğun ən yaxşı kateqoriyanı seç. Heç bir əlavə mətn olmadan, yalnız seçdiyin cavabın hərfini (məsələn, A, B, C və s.) yaz."        # version 1
        #         "content": "Sənə müştəri tərəfindən banka göndərilən sorğu və onunla əlaqəli bir neçə seçim təqdim edilir. Müştəri tərəfindən banka göndərilən sorğunun mövzusunun təqdim edilən kateqoriyalardan hansına aid olduğunu müəyyən et. Cavab olaraq heç bir əlavə mətn olmadan, ancaq doğru variantı qaytar (məsələn: A, B, C və s.)."         # version 1 with Reshad's instruction
        #         "content": "Müştəri tərəfindən banka göndərilən sorğunun mövzusunun təqdim edilən kateqoriyalardan hansına aid olduğunu müəyyən et. Cavab olaraq ancaq doğru variantı qaytar."                # Reshad's own instruction
        #     },
        #    {
        #         "role": "user",
        #         "content": "Sual:\nMənə yaxın filial tapmağa kömək edin.\nSeçimlər:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nCavab:"
        #         # "content": "Sual:\nMənə yaxın filial tapmağa kömək edin.\nSeçimlər:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nCavab:"   # Reshad's updated data
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "J"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nSeçimlər:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nCavab:"
        #         # "content": "Sual:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nSeçimlər:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nCavab:"   # Reshad's updated data
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "G"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nSeçimlər:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nCavab:"
        #         # "content": "Sual:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nSeçimlər:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nCavab:"    # Reshad's updated version
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "D"
        #     },
            # {
            #     "role": "user",
            #     "content": "Sual:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nSeçimlər:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nCavab:"
            #   # "content": "Sual:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nSeçimlər:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nCavab:" # Reshad's updated version
            # },
            # {
            #     "role": "assistant",
            #     "content": "A"
            # },
            # {
            #     "role": "user",
            #     "content": "Sual:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nSeçimlər:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nCavab:"
            #   # "content": "Sual:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nSeçimlər:\nA) BANKOMAT, B) KOMİSSİYA, C) DİGƏR, D) KART, E) HESAB, F) KÖÇÜRMƏ, G) ŞİFRƏ, H) KREDİT, I) ƏLAQƏ, J) TAP\n\nCavab:"    # Reshad's updated version
            # },
            # {
            #     "role": "assistant",
            #     "content": "E"
            # },
        #     {
        #         "role": "user",
        #         "content": f"Sual:\n{question}\nSeçimlər:\n{options}"
        #     }
         # ]


    elif dstype == 'mc': 

        try:
            formatted_options = options.replace(", ", ",\n")
            # formatted_options = options.replace(", ", "\n") # v1.2
            options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nBank krediti haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır\nB) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir\nC) Təminatsız verilə bilər\nD) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nDepozit haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Depozit müqaviləsi 0 (sıfır) faiz dərəcəsi ilə bağlana bilməz\nB) Tələbli və ya müddətli ola bilər\nC) Cari hesabda saxlanılan vəsait də depozit sayılır\nD) Müştərinin tələbi ilə qaytarılmalıdır.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nİpoteka krediti ilə bağlı hansı ifadə doğrudur?\nOptions:\nA) İpoteka krediti mütləq əmlak zəmanəti ilə verilir\nB) İpoteka krediti yalnız maşın almaq üçün istifadə edilə bilər\nC) İpoteka krediti heç bir faiz ödənilmədən verilə bilər\nD) İpoteka krediti yalnız banklar tərəfindən təqdim olunur.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nAşağıdakı müddəalardan hansı yalnışdır\nOptions:\nA) tələbələrə kredit verilə bilər, B) borcalan (gəliri muzdlu işlə əlaqədar olan) ən azı son 6 (altı) ay üzrə aylıq gəlirlərini təsdiq edən sənəd təqdim etməlidir, C) təminatlı aktiv - təminatı məqbul təminat olduqda onun bazar dəyəri təmin etdiyi öhdəlik məbləğinin 100 (yüz) faizindən, təminat məqbul təminat olmadıqda isə 150 (yüz əlli) faizindən az olmayan aktiv, D) Məqbul təminat bankın daxili qaydalarına əsasən müəyyən edilir\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },            
            {
                "role": "user",
                "content": "Question:\nVergi orqani ilə aparılan əməliyyatlarda bankların məsuliyyətinə dair qeyd edilənlərdən biri doğru deyil\nOptions:\nA) Banklar hüquqi şəxslərə, onların filial və nümayəndəliklərinə, fərdi sahibkarlara yalnız onlar vergi orqanı tərəfindən verilən şəhadətnamə-dublikatını təqdim etdikdən sonra bank hesabları açırlar. Şəhadətnamə-dublikat alındığı tarixdən 10 gün müddətində hesabın açılması üçün istifadə olunmadıqda qüvvədən düşmüş hesab edilir., B) Vergi borcunun bank hesabinda dondurulması tələbi gəldikdə bank müştərinin milli və ya xarici valyutada cari və ya digər hesablarında borc məbləğinin 105 faizi həcmində pul vəsaiti üzrə məxaric əməliyyatları məhdudlaşdırır, C) Vergi ödəyicisinin milli və ya xarici valyutada cari və ya digər hesablarında vəsait olduqda vergi orqanının alınmalı sərəncamı bank və ya digər kredit təşkilatı tərəfindən 3(üç) gün ərzində icra edilir., D) Sərəncam gəldiyi zaman müştərinin hesabinda vəsait olmadığı və qismən kifayət etdiyi halda sərəncam bankda saxlanılır, vəsait daxil olduqca icra edilir\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },            
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]

        # aze
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Sən Azərbaycan dilində verilmiş suallara əsaslanaraq ən dəqiq cavabı seçən süni intellektsən. Sənə verilmiş sualdan və seçimlərdən yola çıxaraq, ən uyğun olan tək hərfi (A, B, C, D) seç. Yalnız seçilmiş cavabın hərfini, heç bir əlavə mətni olmadan cavab ver."
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nBank krediti haqqında deyilənlərdən hansı yalnışdır?\nSeçimlər:\nA) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır\nB) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir\nC) Təminatsız verilə bilər\nD) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "D"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nDepozit haqqında deyilənlərdən hansı yalnışdır?\nSeçimlər:\nA) Depozit müqaviləsi 0 (sıfır) faiz dərəcəsi ilə bağlana bilməz\nB) Tələbli və ya müddətli ola bilər\nC) Cari hesabda saxlanılan vəsait də depozit sayılır\nD) Müştərinin tələbi ilə qaytarılmalıdır.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "A"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nİpoteka krediti ilə bağlı hansı ifadə doğrudur?\nSeçimlər:\nA) İpoteka krediti mütləq əmlak zəmanəti ilə verilir\nB) İpoteka krediti yalnız maşın almaq üçün istifadə edilə bilər\nC) İpoteka krediti heç bir faiz ödənilmədən verilə bilər\nD) İpoteka krediti yalnız banklar tərəfindən təqdim olunur.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "A"
        #     },
            # {
            #     "role": "user",
            #     "content": "Sual:\nAşağıdakı müddəalardan hansı yalnışdır\nSeçimlər:\nA) tələbələrə kredit verilə bilər, B) borcalan (gəliri muzdlu işlə əlaqədar olan) ən azı son 6 (altı) ay üzrə aylıq gəlirlərini təsdiq edən sənəd təqdim etməlidir, C) təminatlı aktiv - təminatı məqbul təminat olduqda onun bazar dəyəri təmin etdiyi öhdəlik məbləğinin 100 (yüz) faizindən, təminat məqbul təminat olmadıqda isə 150 (yüz əlli) faizindən az olmayan aktiv, D) Məqbul təminat bankın daxili qaydalarına əsasən müəyyən edilir\n\nCavab:"
            # },
            # {
            #     "role": "assistant",
            #     "content": "D"
            # },            
            # {
            #     "role": "user",
            #     "content": "Sual:\nVergi orqani ilə aparılan əməliyyatlarda bankların məsuliyyətinə dair qeyd edilənlərdən biri doğru deyil\nSeçimlər:\nA) Banklar hüquqi şəxslərə, onların filial və nümayəndəliklərinə, fərdi sahibkarlara yalnız onlar vergi orqanı tərəfindən verilən şəhadətnamə-dublikatını təqdim etdikdən sonra bank hesabları açırlar. Şəhadətnamə-dublikat alındığı tarixdən 10 gün müddətində hesabın açılması üçün istifadə olunmadıqda qüvvədən düşmüş hesab edilir., B) Vergi borcunun bank hesabinda dondurulması tələbi gəldikdə bank müştərinin milli və ya xarici valyutada cari və ya digər hesablarında borc məbləğinin 105 faizi həcmində pul vəsaiti üzrə məxaric əməliyyatları məhdudlaşdırır, C) Vergi ödəyicisinin milli və ya xarici valyutada cari və ya digər hesablarında vəsait olduqda vergi orqanının alınmalı sərəncamı bank və ya digər kredit təşkilatı tərəfindən 3(üç) gün ərzində icra edilir., D) Sərəncam gəldiyi zaman müştərinin hesabinda vəsait olmadığı və qismən kifayət etdiyi halda sərəncam bankda saxlanılır, vəsait daxil olduqca icra edilir\n\nCavab:"
            # },
            # {
            #     "role": "assistant",
            #     "content": "C"
            # },            
        #     {
        #         "role": "user",
        #         "content": f"Sual:\n{question}\nSeçimlər:\n{options}"
        #     }
        # ]



    elif dstype == 'arc':

        try:
            formatted_options = ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
            # formatted_options = '\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)]) # v2
            # print(formatted_options)
            options = formatted_options

        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'
    
        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. You are an AI tasked with selecting the most accurate answer in Azerbaijani based on a given question. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nHansı material ən yaxşı istilik izolyatorudur?\nOptions:\nA) şüşə,\nB) metal,\nC) su,\nD) ağac\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                # "content": "Question:\nDuzları sudan ayırmağın ən yaxşı yolu\nOptions:\nA) yağ,\nB) istilik,\nC) maqnit,\nD) əsəbləşmə alkoqolu\n\nAnswer:" # old version
                "content": "Question:\nDuzun sudan ayrılması üçün ən yaxşı yol ___ istifadəsidir\nOptions:\nA) neft,\nB) İstilik,\nC) maqnit,\nD) İsopropil spirt\n\nAnswer:" # new arc version
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                # "content": "Question:\nTələbə təcrübəni təkrarlaya bilmək üçün hansı addımı atmalıdır?\nOptions:\nA) təcrübəni bir neçə dəfə təkrarlayın,\nB) bütün prosedurların qeydini saxlayır,\nC) Təcrübə quruluşunun bir modelini hazırlayın,\nD) Təcrübə şərtlərinin bir hissəsini dəyişdirin\n\nAnswer:" # old version
               "content": "Question:\nŞagird təcrübəni təkrarlaya bilmək üçün ilk öncə hansı addımı atmalıdır?\nOptions:\nA) Təcrübəni dəfələrlə aparmalıdır,\nB) bütün prosedurları qeydə almalıdır,\nC) Eksperimentin quruluşunun modelini hazırlamalıdır,\nD) Eksperimentin bəzi şərtlərini dəyişməlidir\n\nAnswer:" # new arc
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                # "content": "Question:\nDDT adlı kimyəvi maddə bir vaxtlar zərərli həşəratları öldürmək üçün istifadə olunurdu. Araşdırmalar bu kimyəvi maddənin bəzi quş növləri üçün zərərli olduğunu göstərdikdə, DDT-nin istifadəsi dayandırıldı. Elmi proses DDT-nin quşlara zərər verdiyini anlamalarına necə kömək edə bildi?\nOptions:\nA) Kimyəvi maddə ilə bağlı müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə ilə aparılan testlər fərqli nəticələr verdi,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlardan istifadə edilmişdir\n\nAnswer:" # old version
                "content": "Question:\nDDT adlı bir kimyəvi maddə bir zamanlar zərərverici həşəratları öldürmək üçün istifadə edilirdi. Araşdırmalar bu kimyəvi maddənin bəzi növ quşlar üçün zərərli olduğunu göstərdiyində, DDT-nin istifadəsi dayandırıldı. Elmi proses alimlərin DDT-nin quşlara zərərli olduğunu başa düşmələrinə ən yaxşı şəkildə necə kömək edə bildi?\nOptions:\nA) Kimyəvi maddə haqqında müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə üzərində aparılan testlər müxtəlif nəticələr verdi,\nC) Kimyəvi maddə üzrə təkrarlanan sınaqlar eyni nəticəni göstərdi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlar tətbiq edildi\n\nAnswer:" # new version
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                # "content": "Question:\nKainatdakı bütün qalaktikalar\nOptions:\nA) çoxlu ulduzlardan ibarətdir,\nB) saatın istiqamətində dönür,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) eyni sayda ulduz var\n\nAnswer:"
                "content": "Question:\nSəs dalğaları və okean dalğaları hər ikisi oxşardır çünki, hər ikisi\nOptions:\nA) uzununa dalğalardır,\nB) eyni tezlikdədirlər,\nC) enerjini bir yerdən o biri yerə daşıyır,\nD) Müəyyən vasitəni bir yerdən başqa yerə aparır\n\nAnswer:" # new version
            },
            {
                "role": "assistant",
                "content": "A"
                # "content": "C"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}"
            }
        ]

        # az
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Sən suallara Azərbaycan dilində, məntiq və bilik əsasında (based on reasoning and knowledge) cavab verən süni intellektsən. Verilmiş suala əsaslanaraq ən dəqiq cavabı seçmək sənin tapşırığındır. Suala ən yaxşı cavab verən tək hərfi (A, B, C, D) seç. Yalnız seçilmiş cavabın hərfini, heç bir əlavə mətni olmadan cavab ver."
        #     },
#             {
#                 "role": "user",
#                 "content": "Sual:\nHansı material ən yaxşı istilik izolyatorudur?\nSeçimlər:\nA) şüşə,\nB) metal,\nC) su,\nD) ağac\n\nCavab:"
#             },
#             {
#                 "role": "assistant",
#                 "content": "D"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nDuzları sudan ayırmağın ən yaxşı yolu\nSeçimlər:\nA) yağ,\nB) istilik,\nC) maqnit,\nD) əsəbləşmə alkoqolu\n\nCavab:" # old version
#                 # "content": "Sual:\nDuzun sudan ayrılması üçün ən yaxşı yol ___ istifadəsidir\nSeçimlər:\nA) neft,\nB) İstilik,\nC) maqnit,\nD) İsopropil spirt\n\nCavab:" # new arc version
#             },
#             {
#                 "role": "assistant",
#                 "content": "B"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nTələbə təcrübəni təkrarlaya bilmək üçün hansı addımı atmalıdır?\nSeçimlər:\nA) təcrübəni bir neçə dəfə təkrarlayın,\nB) bütün prosedurların qeydini saxlayır,\nC) Təcrübə quruluşunun bir modelini hazırlayın,\nD) Təcrübə şərtlərinin bir hissəsini dəyişdirin\n\nCavab:" # old version
# #                "content": "Sual:\nŞagird təcrübəni təkrarlaya bilmək üçün ilk öncə hansı addımı atmalıdır?\nSeçimlər:\nA) Təcrübəni dəfələrlə aparmalıdır,\nB) bütün prosedurları qeydə almalıdır,\nC) Eksperimentin quruluşunun modelini hazırlamalıdır,\nD) Eksperimentin bəzi şərtlərini dəyişməlidir\n\nCavab:" # new arc
#             },
#             {
#                 "role": "assistant",
#                 "content": "B"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nDDT adlı kimyəvi maddə bir vaxtlar zərərli həşəratları öldürmək üçün istifadə olunurdu. Araşdırmalar bu kimyəvi maddənin bəzi quş növləri üçün zərərli olduğunu göstərdikdə, DDT-nin istifadəsi dayandırıldı. Elmi proses DDT-nin quşlara zərər verdiyini anlamalarına necə kömək edə bildi?\nSeçimlər:\nA) Kimyəvi maddə ilə bağlı müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə ilə aparılan testlər fərqli nəticələr verdi,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlardan istifadə edilmişdir\n\nCavab:" # old version
#                 # "content": "Sual:\nDDT adlı bir kimyəvi maddə bir zamanlar zərərverici həşəratları öldürmək üçün istifadə edilirdi. Araşdırmalar bu kimyəvi maddənin bəzi növ quşlar üçün zərərli olduğunu göstərdiyində, DDT-nin istifadəsi dayandırıldı. Elmi proses alimlərin DDT-nin quşlara zərərli olduğunu başa düşmələrinə ən yaxşı şəkildə necə kömək edə bildi?\nSeçimlər:\nA) Kimyəvi maddə haqqında müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə üzərində aparılan testlər müxtəlif nəticələr verdi,\nC) Kimyəvi maddə üzrə təkrarlanan sınaqlar eyni nəticəni göstərdi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlar tətbiq edildi\n\nCavab:" # new version
#             },
#             {
#                 "role": "assistant",
#                 "content": "C"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nKainatdakı bütün qalaktikalar\nSeçimlər:\nA) çoxlu ulduzlardan ibarətdir,\nB) saatın istiqamətində dönür,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) eyni sayda ulduz var\n\nCavab:"
#                 # "content": "Sual:\nSəs dalğaları və okean dalğaları hər ikisi oxşardır çünki, hər ikisi\nSeçimlər:\nA) uzununa dalğalardır,\nB) eyni tezlikdədirlər,\nC) enerjini bir yerdən o biri yerə daşıyır,\nD) Müəyyən vasitəni bir yerdən başqa yerə aparır\n\nCavab:"
#             },
#             {
#                 "role": "assistant",
#                 "content": "A"
#                 # "content": "C"
#             },
        # #     {
        #         "role": "user",
        #         "content": f"Sual:\n{question}\nSeçimlər:\n{options}"
        #     }
        # ]



    else:
        raise("Invalid dstype")
    
    answer = ''
    try:
        stream = client_ollama.chat(
            model=model,  
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"Error during streaming: {e}")
        return "Error"

    try:
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                answer += chunk['message']['content']
            else:
                print(f"Unexpected chunk format: {chunk}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        return "Error"
    
    return answer




"""
    Version 1: Compare answers"
"""
def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    # return 1 if actual_answer.lower() == predicted_answer.lower() else 0 # v1
    
    # print("actual_answer:", actual_answer)
    # print("predicted_answer:", predicted_answer)
    
    if pd.notna(predicted_answer) and isinstance(predicted_answer, str) and predicted_answer.strip():
        matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer) # v2
    else:
        return 0

    if predicted_answer.lower() == "answer":
        return 0

    if matched_predicted and matched_predicted.group(1) is not None and matched_predicted.group(1) != "":
        # print("matched_predicted:", matched_predicted.group(1).lower())
        # print('\n\n')
    
        return 1 if actual_answer.lower() == matched_predicted.group(1).lower() else 0
    else:
        return 0



"""
    Version 2: Compare answers and write the result to a file."
"""
def compare_answers_and_save(actual_answer: str, predicted_answer: str, benchmark_type: str, model_name: str) -> int:
    """
    Compare the actual answer with the predicted answer and log results to a file.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    - benchmark_type (str): Type of benchmark.
    - model_name (str): Name of the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    results = []
    
    # print("actual_answer:", actual_answer)
    # print("predicted_answer:", predicted_answer)

    if pd.notna(predicted_answer) and isinstance(predicted_answer, str) and predicted_answer.strip():
        matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer)
    else:
        results.append([actual_answer, predicted_answer, None, 0])
        return 0

    if predicted_answer.lower() == "answer":
        results.append([actual_answer, predicted_answer, None, 0])
        return 0

    if matched_predicted and matched_predicted.group(1) is not None:
        matched_predicted_value = matched_predicted.group(1).lower()
        # print("matched_predicted:", matched_predicted_value)
        # print('\n\n')
        
        result = 1 if actual_answer.lower() == matched_predicted_value else 0
        results.append([actual_answer, predicted_answer, matched_predicted_value, result])
        save_to_file(benchmark_type, model_name, results)
        return result
    else:
        results.append([actual_answer, predicted_answer, None, 0])
        save_to_file(benchmark_type, model_name, results)
        return 0


def save_to_file(benchmark_type: str, model_name: str, results: list):

    # Create the evaluation_scores directory if it doesn't exist
    evaluation_scores_dir = "evaluation_scores"
    os.makedirs(evaluation_scores_dir, exist_ok=True)

    df = pd.DataFrame(results, columns=["Actual Answer", "Predicted Answer", "Matched Predicted", "Score"])

    filename_csv = os.path.join(evaluation_scores_dir, f"{benchmark_type}_{model_name}_evaluation_scores.csv")


    # # Append to the CSV file if it exists; otherwise, create a new one # v1
    # df.to_csv(filename_csv, mode="a", header=not os.path.exists(filename_csv), index=False, encoding='utf-8-sig')


    if os.path.exists(filename_csv): # v2
        existing_df = pd.read_csv(filename_csv)

        if existing_df.empty:
            # If the file exists but is empty, write with header
            df.to_csv(filename_csv, mode="w", header=True, index=False, encoding='utf-8-sig')
        else:
            # If the file exists and has rows, append without header
            df.to_csv(filename_csv, mode="a", header=False, index=False, encoding='utf-8-sig')
    else:
        # If the file does not exist, create it with header
        df.to_csv(filename_csv, mode="w", header=True, index=False, encoding='utf-8-sig')


