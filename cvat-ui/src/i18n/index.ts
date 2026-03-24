import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import zh from './locales/zh.json';

const savedLng = localStorage.getItem('cvat-lang') || 'en';

i18n
    .use(initReactI18next)
    .init({
        lng: savedLng,
        fallbackLng: 'en',
        interpolation: { escapeValue: false },
        resources: {
            en: { translation: en },
            zh: { translation: zh },
        },
    })
    .then(() => { console.log('i18n init successfully!'); })
    .catch((err) => { console.error('i18n init failed!', err); });
