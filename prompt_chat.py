#!/usr/bin/env python3
"""
Скрипт для работы с промптами из папки prompts/
Читает JSON файлы с промптами, принимает вопрос пользователя и отправляет в OpenAI
"""

import os
import json
import glob
from openai import OpenAI
from dotenv import load_dotenv

#комментарий
def load_prompts_from_folder(prompts_folder='prompts'):
    """Загружает все промпты из папки prompts/"""
    prompts = []
    
    if not os.path.exists(prompts_folder):
        print(f"❌ Ошибка: Папка '{prompts_folder}' не найдена")
        return None
    
    # Ищем все JSON файлы в папке prompts
    json_files = glob.glob(os.path.join(prompts_folder, '*.json'))
    
    if not json_files:
        print(f"❌ Ошибка: В папке '{prompts_folder}' не найдено JSON файлов")
        return None
    
    # Загружаем каждый файл
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts.append(data)
        except json.JSONDecodeError as e:
            print(f"⚠️  Предупреждение: Не удалось загрузить {file_path}: {e}")
            continue
        except Exception as e:
            print(f"⚠️  Предупреждение: Ошибка при загрузке {file_path}: {e}")
            continue
    
    return prompts


def display_prompts(prompts):
    """Отображает список доступных промптов"""
    print("\n📋 Доступные промпты:")
    print("=" * 80)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n{idx}. {prompt.get('name', 'Без названия')}")
        print(f"   📌 ID: {prompt.get('prompt_id', 'N/A')}")
        print(f"   📂 Категория: {prompt.get('category', 'N/A')}")
        print(f"   ℹ️  Описание: {prompt.get('description', 'N/A')}")
        
        # Показываем роль (первые 100 символов)
        role = prompt.get('role', '')
        if role:
            print(f"   🎭 Роль: {role[:100]}{'...' if len(role) > 100 else ''}")
        
        # Показываем контекст (первые 100 символов)
        context = prompt.get('context', '')
        if context:
            print(f"   📝 Контекст: {context[:100]}{'...' if len(context) > 100 else ''}")
        
        if 'test_input' in prompt:
            print(f"   ✨ Есть тестовый пример")
    
    print("=" * 80)


def select_prompt(prompts):
    """Позволяет пользователю выбрать промпт"""
    while True:
        try:
            choice = input(f"\n🔢 Выберите промпт (1-{len(prompts)}) или 'выход' для завершения: ").strip()
            
            if choice.lower() in ['выход', 'exit', 'quit', 'q', 'стоп', 'stop']:
                return None
            
            choice = int(choice)
            
            if 1 <= choice <= len(prompts):
                return prompts[choice - 1]
            else:
                print(f"❌ Пожалуйста, введите число от 1 до {len(prompts)}")
        except ValueError:
            print("❌ Пожалуйста, введите корректное число или 'выход'")


def get_user_question(prompt):
    """Получает вопрос пользователя"""
    # Проверяем есть ли тестовый вопрос
    if 'test_input' in prompt and prompt['test_input']:
        print(f"\n💡 Доступен тестовый вопрос:")
        print(f"   {prompt['test_input']}")
        use_test = input("\n🤔 Использовать тестовый вопрос? (y/n, по умолчанию n): ").strip().lower()
        
        if use_test in ['y', 'yes', 'да', 'д']:
            return prompt['test_input']
    
    # Получаем свой вопрос
    question = input("\n💬 Введите ваш вопрос: ").strip()
    
    if not question:
        print("❌ Вопрос не может быть пустым")
        return None
    
    return question


def get_model_params():
    """Получает параметры модели от пользователя"""
    print("\n⚙️  Настройки модели:")
    
    # Temperature
    try:
        temp_input = input("🌡️  Введите temperature (0.0-1.0, по умолчанию 0.7): ").strip()
        temperature = float(temp_input) if temp_input else 0.7
        if not 0.0 <= temperature <= 1.0:
            print("❌ Temperature должен быть от 0.0 до 1.0, используем 0.7")
            temperature = 0.7
    except ValueError:
        print("❌ Некорректное значение, используем 0.7")
        temperature = 0.7
    
    # Max tokens
    try:
        tokens_input = input("🔢 Введите max_tokens (по умолчанию 2000): ").strip()
        max_tokens = int(tokens_input) if tokens_input else 2000
        if max_tokens <= 0:
            print("❌ max_tokens должен быть больше 0, используем 2000")
            max_tokens = 2000
    except ValueError:
        print("❌ Некорректное значение, используем 2000")
        max_tokens = 2000
    
    # Model
    model = input("🤖 Введите модель (по умолчанию gpt-4o-mini): ").strip()
    if not model:
        model = "gpt-4o-mini"
    
    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'model': model
    }


def build_message(prompt, user_question):
    """Строит сообщение для OpenAI API"""
    system_message = prompt.get('role', 'Вы - полезный ассистент')
    
    # Собираем context если есть
    context = prompt.get('context', '')
    
    # Формируем user message
    user_message = f"Контекст: {context}\n\n" if context else ""
    
    # Добавляем описание задачи
    if 'description' in prompt:
        user_message += f"Описание задачи: {prompt['description']}\n\n"
    
    # Добавляем информацию о структуре вывода если есть
    if 'structure' in prompt:
        user_message += "Структура ответа:\n"
        if 'components' in prompt['structure']:
            for component in prompt['structure']['components']:
                user_message += f"- {component.get('name', '')}: {component.get('description', '')}\n"
        user_message += "\n"
    
    # Добавляем форматирование если есть
    if 'format' in prompt:
        user_message += "Требования к формату ответа:\n"
        format_info = prompt['format']
        for key, value in format_info.items():
            if key != 'requirements' and not isinstance(value, list):
                user_message += f"- {key}: {value}\n"
        user_message += "\n"
    
    # Добавляем пользовательский вопрос
    user_message += f"Вопрос пользователя:\n{user_question}"
    
    return system_message, user_message


def send_to_openai(client, system_message, user_message, params):
    """Отправляет запрос в OpenAI"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    print("\n⏳ Отправляем запрос к OpenAI...")
    
    try:
        response = client.chat.completions.create(
            model=params['model'],
            messages=messages,
            temperature=params['temperature'],
            max_tokens=params['max_tokens']
        )
        
        return response
    except Exception as e:
        print(f"❌ Ошибка при отправке запроса: {e}")
        return None


def display_response(response, prompt_name):
    """Отображает ответ от OpenAI"""
    if not response:
        return
    
    print("\n" + "=" * 80)
    print(f"✅ Ответ от OpenAI - {prompt_name}")
    print("=" * 80)
    print(response.choices[0].message.content)
    print("=" * 80)
    
    # Статистика
    print(f"\n📊 Информация о запросе:")
    print(f"   • Модель: {response.model}")
    print(f"   • Использовано токенов: {response.usage.total_tokens}")
    print(f"   • Промпт токены: {response.usage.prompt_tokens}")
    print(f"   • Ответ токены: {response.usage.completion_tokens}")


def main():
    # Загружаем переменные окружения
    load_dotenv()
    
    # Получаем API ключ
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Ошибка: Не найден OPENAI_API_KEY в .env файле")
        print("📝 Создайте файл .env на основе env_example.txt и добавьте ваш API ключ")
        return
    
    # Инициализируем клиент
    client = OpenAI(api_key=api_key)
    
    print("\n🤖 OpenAI API - Работа с промптами")
    print("=" * 80)
    
    try:
        # Загружаем промпты
        prompts = load_prompts_from_folder('prompts')
        if not prompts:
            return
        
        print(f"✅ Загружено промптов: {len(prompts)}")
        
        # Показываем промпты
        display_prompts(prompts)
        
        # Выбираем промпт
        selected_prompt = select_prompt(prompts)
        if not selected_prompt:
            print("\n👋 До свидания!")
            return
        
        print(f"\n✅ Выбран промпт: {selected_prompt.get('name', 'N/A')}")
        
        # Получаем вопрос пользователя
        user_question = get_user_question(selected_prompt)
        if not user_question:
            return
        
        # Получаем параметры модели
        params = get_model_params()
        
        # Формируем сообщения
        system_message, user_message = build_message(selected_prompt, user_question)
        
        # Отправляем в OpenAI
        response = send_to_openai(client, system_message, user_message, params)
        
        if response:
            display_response(response, selected_prompt.get('name', 'N/A'))
        
        print("\n👋 Готово! До свидания!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
