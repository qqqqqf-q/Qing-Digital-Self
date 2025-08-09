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
              { text: '1. QQ数据库的获取', link: '/guide/qq-database' },
              { text: '1.5 (可选) 从视频/音频文件中获取聊天数据', link: '/guide/media-chat-data' },
              { text: '2. 清洗数据', link: '/guide/clean-data' },
              { text: '3. 混合数据', link: '/guide/mix-data' },
              { text: '4. 准备模型', link: '/guide/prepare-model' },
              { text: '5. 微调模型', link: '/guide/fine-tune-model' },
              { text: '6. (建议跳过)微调后直接运行全量模型', link: '/guide/run-full-model' },
              { text: '7. 转换GUFF和量化模型', link: '/guide/convert-model' },
              { text: '8. 运行模型', link: '/guide/run-model' },
              { text: '番外篇: 微调OpenAI OSS模型', link: '/guide/fine-tune-openai-oss-model' }
            ]
          },
          {
            text: '补充',
            items: [
              { text: '修改Logger的语言', link: '/guide/change-logger-language' }
            ]
          },
          {
            text: '总结',
            items: [
              { text: '9. 总结', link: '/guide/summary' }
            ]
          }
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/qqqqqf-q/Qing-Digital-Self' }
        ]
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
              { text: '1. Get QQ Database', link: '/en/guide/qq-database' },
              { text: '1.5 (Optional) Get Chat Data from Media', link: '/en/guide/media-chat-data' },
              { text: '2. Clean Data', link: '/en/guide/clean-data' },
              { text: '3. Mix Data', link: '/en/guide/mix-data' },
              { text: '4. Prepare Model', link: '/en/guide/prepare-model' },
              { text: '5. Fine-tune Model', link: '/en/guide/fine-tune-model' },
              { text: '6. (Optional) Run Full Model', link: '/en/guide/run-full-model' },
              { text: '7. Convert GGUF and Quantize', link: '/en/guide/convert-model' },
              { text: '8. Run Model', link: '/en/guide/run-model' },
              { text: '9. (Extra) Fine-tune OpenAI OSS Model', link: '/en/guide/fine-tune-openai-oss-model' }

            ]
          },
          {
            text: 'Extra',
            items: [
              { text: 'Change Logger Language', link: '/en/guide/change-logger-language' }
            ]
          },

          {
            text: 'Summary',
            items: [
              { text: '9. Summary', link: '/en/guide/summary' }
            ]
          }
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/qqqqqf-q/Qing-Digital-Self' }
        ]
      }
    }
  },
  
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    socialLinks: [
      { icon: 'github', link: 'https://github.com/qqqqqf-q/Qing-Digital-Self' }
    ]
  }
})
