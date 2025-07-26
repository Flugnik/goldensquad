#!/usr/bin/env node
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');

// Проверка API ключа
const apiKey = process.env.GEMINI_API_KEY || 'AIzaSyA1C9K8pXOfkbjJmwFKIrh38GcB1QFF9Qo';
if (!apiKey) {
    console.error('❌ API ключ не найден');
    process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);

async function generateContent(prompt, options = {}) {
    try {
        const model = genAI.getGenerativeModel({ 
            model: options.model || "gemini-pro",
            generationConfig: {
                temperature: options.temperature || 0.8,
                topK: options.topK || 40,
                topP: options.topP || 0.95,
                maxOutputTokens: options.maxTokens || 2048,
            }
        });

        console.log('🤖 Gemini генерирует контент...');
        const result = await model.generateContent(prompt);
        const response = await result.response;
        
        console.log('\n✅ Результат генерации:');
        console.log('=' * 50);
        console.log(response.text());
        console.log('=' * 50);
        
        return response.text();
    } catch (error) {
        console.error('❌ Ошибка генерации:', error.message);
        if (error.message.includes('API_KEY')) {
            console.error('💡 Проверьте правильность API ключа');
        }
        process.exit(1);
    }
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.log(`
🤖 Gemini CLI для проекта "Золотая команда"

Использование:
  node gemini-cli.js "ваш промпт"
  
Примеры:
  node gemini-cli.js "Создай пост про зимнее кормление свиней"
  node gemini-cli.js "Напиши заголовок для поста про выбор поросят"
  
Переменные окружения:
  GEMINI_API_KEY - ключ Google AI API
        `);
        return;
    }
    
    const prompt = args.join(' ');
    await generateContent(prompt);
}

if (require.main === module) {
    main();
}

module.exports = { generateContent };
