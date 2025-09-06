import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/Qing-Digital-Self/',
  locales: {
    root: {
      label: '中文',
      lang: 'zh-CN',
      title: "Qing-Digital-Self",
      description: "清凤的数字分身,并且包含了搭建教程",
      themeConfig: {
        nav: [
          { text: '开始', link: '/' },
          { text: '快速上手', link: '/guide/index' }
        ],
        sidebar: [
          {
            text: '开始',
            items: [
              { text: '简介', link: '/guide/index' },
            ]
          },
          {
            text: '快速上手',
            items: [
              { text: '1. QQ/TG/WX其他数据的获取', link: '/guide/prepare-data' },
              { text: '2. 清洗数据', link: '/guide/clean-data' },
              { text: '3. (可选) 混合数据', link: '/guide/mix-data' },
              { text: '4. 微调模型', link: '/guide/finetune-llama-factory' },
              { text: '4. (请跳过)微调模型(Old)', link: '/guide/fine-tune-model' },
              { text: '5. (建议跳过)微调后直接运行全量模型', link: '/guide/run-full-model' },
              { text: '6. 转换GUFF和量化模型', link: '/guide/convert-model' },
              { text: '7. 运行模型', link: '/guide/run-model' },

            ]
          },
          {
            text: '补充',
            items: [
              { text: '修改Logger的语言', link: '/guide/change-logger-language' },
              { text: '不同种类模型的微调经验(例如Qwen,Llama,Gemma等)', link: '/guide/fine-tune-model-exp' },
              { text: '节省显存', link: '/guide/save-vram' }

            ]
          },

          {
            text: '总结',
            items: [
              { text: '8. 总结', link: '/guide/summary' }
            ]
          }
        ],
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: "Qing-Digital-Self",
      description: "Qing's digital avatar with complete setup tutorial",
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Quick Start', link: '/en/guide/index' }
        ],
        sidebar: [
          {
            text: 'Getting Started',
            items: [
              { text: 'Introduction', link: '/en/guide/index' },
            ]
          },
          {
            text: 'Quick Start',
            items: [
              { text: '1. QQ/TG/Other Data Acquisition', link: '/en/guide/prepare-data' },
              { text: '2. Clean Data', link: '/en/guide/clean-data' },
              { text: '3. (Optional) Mix Data', link: '/en/guide/mix-data' },
              { text: '4. Prepare Model', link: '/en/guide/prepare-model' },
              { text: '5. Fine-tune Model', link: '/en/guide/fine-tune-model' },
              { text: '6. (Recommended to Skip) Run Full Model After Fine-tuning', link: '/en/guide/run-full-model' },
              { text: '7. Convert GGUF and Quantize Model', link: '/en/guide/convert-model' },
              { text: '8. Run Model', link: '/en/guide/run-model' },

            ]
          },
          {
            text: 'Extra',
            items: [
              { text: 'Change Logger Language', link: '/en/guide/change-logger-language' },
              { text: 'Fine-tuning Experience with Different Models (Qwen, Llama, Gemma, etc.)', link: '/en/guide/fine-tune-model-exp' },
              { text: 'VRAM Optimization', link: '/en/guide/save-vram' }
            ]
          },

          {
            text: 'Summary',
            items: [
              { text: '9. summary', link: '/en/guide/summary' }
            ]
          }
        ],
      }
    }
  },
  
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    socialLinks: [
      { icon: 'github', link: 'https://github.com/qqqqqf-q/Qing-Digital-Self' },
      { icon: { svg: `<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>X</title><path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 7.184L18.901 1.153Zm-1.653 19.57h2.608L6.852 3.24H4.21l13.038 17.484Z"/></svg>` }, link: 'https.x.com/qqqqqf5' }
    ]
  }
})