import telegram
from telegram import Poll, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
import random
import json
import torch, transformers
from torch import nn
from transformers import AutoTokenizer
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    PollAnswerHandler,
    PollHandler,
    ConversationHandler,
    CallbackContext
)


import logging
import sqlite3
import re

from transformers import AutoTokenizer, EncoderDecoderModel
import pandas as pd
import sqlite3

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


logger = logging.getLogger(__name__)

ANSWER, TITLE, TAG, TEXTAG, MORE, VAR = range(6)


def prediction(article_text, model_name = "/Users/ico/Desktop/проект прога/rubert_telegram_headlines") -> object:
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, do_basic_tokenize=False, strip_accents=False)
    model = EncoderDecoderModel.from_pretrained(model_name)
    input_ids = tokenizer(
        [article_text],
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    def gusev_model(input_ids, mode):
        if mode == 'inst':
            max_len = 10
        else:
            max_len = 40
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_len,
            no_repeat_ngram_size=3,
            num_beams=10,
            top_p=0.95
        )[0]
        return output_ids

    headline_inst = tokenizer.decode(gusev_model(input_ids, 'inst'), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    headline_news = tokenizer.decode(gusev_model(input_ids, 'news'), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    return headline_inst, headline_news

from russian_paraphrasers import GPTParaphraser
import difflib
import re

paraphraser = GPTParaphraser(model_name="gpt3", range_cand=True, make_eval=False)
def model_sber(text):
    def sber_model(origin, mode):
        if mode == 'inst':
          max_len = 40
        else:
          max_len = 100
        results = paraphraser.generate(
                origin, n=10, temperature=1,
                top_k=10, top_p=0.9,
                max_length=max_len, repetition_penalty=1.5,
                threshold=0.8
            )
        try:
            res = results["results"][0]["best_candidates"][0]
            if res.endswith("?"):
                res = results["results"][0]["best_candidates"][1]
        except IndexError:
                res = results["results"][0]["predictions"][0]
        return res
    return sber_model(text, 'inst').upper(), sber_model(text, 'news')


def tag(text_trial, num_tags):
    global main_ans
    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dict_le = {0: 'Coцсети',
               1: 'Авто',
               2: 'Автобизнес',
               3: 'Белоруссия',
               4: 'Бизнес',
               5: 'Вещи',
               6: 'Гаджеты',
               7: 'Город',
               8: 'Госэкономика',
               9: 'Дача',
               10: 'Движение',
               11: 'Деньги',
               12: 'Достижения',
               13: 'Еда',
               14: 'Жизнь',
               15: 'Закавказье',
               16: 'Звери',
               17: 'Игры',
               18: 'Инструменты',
               19: 'Интернет',
               20: 'Искусство',
               21: 'История',
               22: 'Квартира',
               23: 'Киберпреступность',
               24: 'Кино',
               25: 'Книги',
               26: 'Конфликты',
               27: 'Космос',
               28: 'Криминал',
               29: 'Люди',
               30: 'Мемы',
               31: 'Мир',
               32: 'Мнения',
               33: 'Молдавия',
               34: 'Москва',
               35: 'Музыка',
               36: 'Наука',
               37: 'Общество',
               38: 'Оружие',
               39: 'Офис',
               40: 'Политика',
               41: 'Пресса',
               42: 'Преступность',
               43: 'Прибалтика',
               44: 'Происшествия',
               45: 'Регионы',
               46: 'Реклама',
               47: 'Россия',
               48: 'Рынки',
               49: 'События',
               50: 'Софт',
               51: 'Стиль',
               52: 'Театр',
               53: 'Техника',
               54: 'Туризм',
               55: 'Украина',
               56: 'Футбол',
               57: 'Хоккей',
               58: 'Часы',
               59: 'Явления'}

    # загружаем первую модель
    class BeastModel_1(nn.Module):
        def __init__(self, seq_len, vocab_size, hidden_sizes=[100, 64, 128, 256, 256, 64, 60]):
            self.hidden_sizes = hidden_sizes
            self.seq_len = seq_len
            super().__init__()

            self.embed = nn.Embedding(vocab_size, self.hidden_sizes[0])

            self.lstm = nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[1], num_layers=2, batch_first=True,
                                bidirectional=True)

            self.fc_1 = nn.Sequential(nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]), nn.LeakyReLU())
            self.fc_2 = nn.Sequential(nn.Linear(self.hidden_sizes[3], self.hidden_sizes[4]), nn.LeakyReLU())
            self.fc_3 = nn.Sequential(nn.Linear(self.hidden_sizes[4], self.hidden_sizes[5]), nn.LeakyReLU())
            self.fc_4 = nn.Sequential(nn.Linear(self.hidden_sizes[5], self.hidden_sizes[6]))

            self.linear_layers = nn.Sequential(self.fc_1, self.fc_2, self.fc_3, self.fc_4)

        def forward(self, x):
            x = self.embed(x)
            x = self.lstm(x)[1][1]
            x = x.view(x.shape[0] // 2, 2, x.shape[1], x.shape[2])
            x = x[1].permute(1, 0, 2).flatten(1)

            return self.linear_layers(x)


    model = BeastModel_1(300, len(tokenizer.vocab))
    model.load_state_dict(torch.load('/Users/ico/Desktop/проект прога/model2/model.pt', map_location='cpu'))


    # как принимаем на вход текст

    def tokenize_text(examples, def_tokenizer):
        tokenized_inputs = def_tokenizer(examples, truncation=True, max_length=300)['input_ids']
        return tokenized_inputs


    max_length = 300


    def preprocess_text_2(text):
        tokens_left = text
        if len(tokens_left) >= max_length:
            return tokens_left[:max_length]
        else:
            pads = [0] * (max_length - len(tokens_left))
            return tokens_left + pads


    input_text = preprocess_text_2(tokenize_text(text_trial, tokenizer))

    # делаем само предсказание
    # dict_le_get отдельным файлом пришлю

    _, indices = torch.topk(model(torch.tensor(input_text).resize_((1, 1))), k=num_tags, dim=1)
    ans = []
    for ind in indices.tolist()[0]:
        ans.append(dict_le.get(ind))
        myString = ' #'.join(ans)
        main_ans = 'Список рекомендуемых тэгов: #' + myString.lower()
    return main_ans


def start(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [['Заголовок','Тэг']]
    update.message.reply_text(
        'Привет! Выбери то, что нужно для твоего текста!',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return ANSWER

def received_answer(update: Update, context: CallbackContext):
    text_1 = update.message.text
    if text_1 == 'Заголовок':
        update.message.reply_text(
        'Введи, пожалуйста, текст!'
        )
        return TITLE
    if text_1 == 'Тэг':
        update.message.reply_text(
        'Введи, пожалуйста, текст!'
        )
        return TEXTAG



def get_headline(update: Update, context: CallbackContext):
    update.message.reply_text(
        'Придется чуть-чуть подождать...'
    )
    text = update.message.text
    headline = prediction(text)
    update.message.reply_text('Первый вариант:' + '\n' + 'Инстаграм-заголовок: ' + headline[0] + '\n\n' + 'Заголовок для новостей: ' + headline[1])

    headline_1 = model_sber(text)
    update.message.reply_text('Второй вариант:' + '\n' + 'Инстаграм-заголовок: ' + headline_1[0] + '\n\n' + 'Заголовок для новостей: ' + headline_1[1])
    reply_keyboard = [['Первый вариант', 'Второй вариант']]
    update.message.reply_text(
        'Выбери вариант заголовка, который тебе понравился больше остальных!',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return VAR

def best_var(update: Update, context: CallbackContext):
    text = update.message.text
    conn = sqlite3.connect('titles.db')
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS best_titles(Лучший заголовок text)")
    c.execute("INSERT INTO best_titles VALUES (?)", (text,))
    reply_keyboard = [['/start', '/help', '/end']]
    update.message.reply_text(
    'Если хочешь получить еще один заголовок или тэг - выбери команду /start. Подробнее о боте по команде /help',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return MORE

def get_text_tag(update: Update, context: CallbackContext) -> str:
    text = update.message.text
    context.user_data['text'] = text
    update.message.reply_text(
        'Введи нужное количество тэгов (от 1 до 10)'
    )
    return TAG

def get_tag(update: Update, context: CallbackContext):
    update.message.reply_text(
        'Придется чуть-чуть подождать...'
    )
    num_tags = update.message.text
    main_tag = tag(context.user_data['text'], int(num_tags))
    update.message.reply_text(main_tag)
    reply_keyboard = [['/start', '/help', '/end']]
    update.message.reply_text(
        'Если хочешь получить еще один заголовок или тэг - выбери команду /start. Подробнее о боте по команде /help',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )
    return MORE

def end(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
		'Хорошо! До скорых встреч!'
	)
    return ConversationHandler.END



# команда help - как работает код
def help(update: Update, context: CallbackContext):
    text = "Бот, который генерирует заголовки и тэги для ваших текстов.\n\
	*Доступные команды:*\n\
	/start запустить бота\n\
	/end остановить бота"
    context.bot.send_message(chat_id=update.effective_chat.id,  parse_mode = 'Markdown', text=text)


def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Прости, я не понимаю, что ты говоришь")


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():

    updater = Updater(token='1807195069:AAFthSXsMrDa46lzdnXF50G8GmPIlTcNVnA', use_context=True)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
		entry_points=[CommandHandler('start', start)],
		states={
			ANSWER: [MessageHandler(Filters.regex('[А-Я][а-я]+'), received_answer)],
			TITLE: [MessageHandler(Filters.language('ru'), get_headline)],
            TEXTAG: [MessageHandler(Filters.regex('[А-Я][а-я]+'), get_text_tag)],
            TAG: [MessageHandler(Filters.regex('[0-9]*'), get_tag)],
            VAR: [MessageHandler(Filters.regex('.+'), best_var)],
			MORE: [
				CommandHandler('start', start),
				CommandHandler('end', end),
				CommandHandler('help', help)
			],
		},
		fallbacks=[CommandHandler('cancel', start),
				   CommandHandler('help', help)],
	)
    dispatcher.add_handler(conv_handler)

    # log all errors
    dispatcher.add_error_handler(error)

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
