
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

# Конфигурация
TOKEN = os.getenv('TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
DEFAULT_SESSION_ID = os.getenv('DEFAULT_SESSION_ID')

# Инициализация LLM (Mistral)
def setup_llm_chain():
    chat_history = InMemoryChatMessageHistory()
    
    messages = [
        ("system", "Ты эксперт по магистерским программам ИТМО. Отвечай кратко и точно. Используй данные с сайтов ИТМО."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.3,
        mistral_api_key=MISTRAL_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="history"
    )

# Парсинг данных с сайтов ИТМО
def fetch_itmo_data():
    urls = [
        'https://abit.itmo.ru/program/master/ai',
        'https://abit.itmo.ru/program/master/ai_product'
    ]
    data = {}
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            data[url] = text[:5000]  # Ограничиваем объем
        except Exception as e:
            data[url] = f"Ошибка: {str(e)}"
    return data


# Обработчики Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот-консультант по магистратурам ИТМО. "
        "Задай вопрос о программах AI или AI Product."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    llm_chain = context.bot_data['llm_chain']
    itmo_data = context.bot_data['itmo_data']
    
    # Добавляем данные ИТМО в контекст вопроса
    augmented_question = f"""
    Вопрос: {user_question}
    Контекст: {itmo_data}
    Ты — ассистент для абитуриентов магистратуры ИТМО. Твоя задача помогать выбирать подходящие программы и рекомендовать дисциплины на основе бэкграунда студента.

Правила общения:
1. Отвечай кратко и по делу
2. Задавай уточняющие вопросы, если информации недостаточно
3. Используй только официальные данные с сайта ИТМО 
4. Для рекомендаций учитывай опыт и интересы абитуриента
5. Если пользователь исользует нецензурную лексику или задает вопросы не по теме - отвечай, что  можешь говорить только о теме поступления в ВУЗ
Шаблон ответа:
1. Сначала уточни интересы (например: "Какое направление вас интересует: AI или AI Product?")
2. Затем спроси про бэкграунд (например: "Какой у вас опыт в программировании?")
3. Дай рекомендации по выборным дисциплинам (например: "Для вашего уровня рекомендую курсы: 1) Продвинутый Python, 2) Основы ML")

Доступные программы:
- Магистратура по AI (ссылка)
- Магистратура по AI Product (ссылка)

Пример диалога:
Абитуриент: Хочу поступить в магистратуру
Ты: Какое направление вас интересует: Artificial Intelligence или AI Product Development?

Абитуриент: AI
Ты: Какой у вас опыт в программировании и machine learning? 

Абитуриент: 2 года Python, базовый ML
Ты: Рекомендую следующие курсы: 
1. Углубленный Machine Learning
2. Нейронные сети и Deep Learning
3. Обработка естественного языка

    """
    
    try:
        response = await llm_chain.ainvoke(
            {"question": augmented_question},
            config={"configurable": {"session_id": DEFAULT_SESSION_ID}}
        )
        await update.message.reply_text(response[:4000])  # Обрезаем длинные ответы
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Запуск бота
def main():
    app = Application.builder().token(TOKEN).build()
    
    # Инициализация данных и LLM
    app.bot_data['itmo_data'] = fetch_itmo_data()
    app.bot_data['llm_chain'] = setup_llm_chain()
    
    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()