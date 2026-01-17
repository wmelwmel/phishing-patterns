from constants import DATA_DIR

# Пример 1: Указываем директорию, в которой содержатся .eml файлы
# (будут обрбаатываться все .eml, включая вложенные подпапки)
messages_to_process = DATA_DIR / "inference_emls"

# Пример 2: Указываем путь до конкретного .eml файла
# messages_to_process = DATA_DIR / "inference_emls" / "1751387580_4412.eml"

# Пример 3: Указываем список строк
# messages_to_process = ["Привет, user! Это первое сообщение.",
#                        "Second sentence to check click here [link]."]

run_id = "ae0977e07a2f4f1d831ab2a66a09dfcc"  # run_id из нашего Mlflow, откуда хотим брать модель
languages = ["en", "ru"]
use_gpu = False
use_txt_split = False  # можно пробовать выставлять True, иногда так письмо разделяется лучше
verbose = False  # дополнительно логгировать/не логгировать результаты в консоль
