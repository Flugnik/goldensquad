const { GoogleGenerativeAI } = require('@google/generative-ai');

const apiKey = 'AIzaSyA1C9K8pXOfkbjJmwFKIrh38GcB1QFF9Qo';
const genAI = new GoogleGenerativeAI(apiKey);

async function listAvailableModels() {
    try {
        console.log('🔍 Загружаю список доступных моделей...');
        
        const models = await genAI.listModels();
        
        console.log('📋 Доступные модели:');
        console.log('=' * 50);
        
        models.forEach((model, index) => {
            console.log(`${index + 1}. ${model.name}`);
            if (model.description) {
                console.log(`   Описание: ${model.description}`);
            }
            console.log('');
        });
        
    } catch (error) {
        console.error('❌ Ошибка получения моделей:', error.message);
        
        // Альтернативный способ - попробуем разные модели
        console.log('\n🔄 Пробуем популярные модели:');
        const testModels = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-1.0-pro',
            'models/gemini-pro',
            'models/gemini-1.5-flash'
        ];
        
        for (const modelName of testModels) {
            try {
                const model = genAI.getGenerativeModel({ model: modelName });
                const result = await model.generateContent('Тест');
                console.log(`✅ Модель ${modelName} - РАБОТАЕТ`);
                break;
            } catch (err) {
                console.log(`❌ Модель ${modelName} - не доступна`);
            }
        }
    }
}

listAvailableModels();
