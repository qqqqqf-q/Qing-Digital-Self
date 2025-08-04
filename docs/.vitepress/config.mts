import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Qing-Digital-Self",
  description: "清凤的数字分身,并且包含了搭建教程",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '开始', link: '/' },
      { text: '快速上手', link: '/guide/index.md' }
    ],

    sidebar: [
      {
        text: '开始',
        items: [
          { text: '简介', link: '/guide/index.md' },
        ]
      },
      {
        text: '快速上手',
        items: [
          { text: '1. QQ数据库的获取', link: '/guide/qq-database.md' },
          { text: '2. 清洗数据', link: '/guide/clean-data.md' },
          { text: '3. 混合数据', link: '/guide/mix-data.md' },
          { text: '4. 准备模型', link: '/guide/prepare-model.md' },
          { text: '5. 微调模型', link: '/guide/fine-tune-model.md' },
          { text: '6. (建议跳过)微调后直接运行全量模型', link: '/guide/run-full-model.md' },
          { text: '7. 转换GUFF和量化模型', link: '/guide/convert-model.md' },
          { text: '8. 运行模型', link: '/guide/run-model.md' }
        ]
      },
      {
        text: '总结',
        items: [
          { text: '9. 总结', link: '/guide/summary.md' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/qqqqqf-q/Qing-Digital-Self' }
    ]
  }
})
