## Original Idea:
建立一个中医理疗领域的短视频营销平台，能够实时感知当下热点关键字，并智能地与中医热门知识库结合，利用人工智能的文生视频工具，自动生成新的短视频，
自动上传到多个短视频平台的多个账号，并自动从平台抓取它们的评价反馈,自动进行大量数据的正面和负面评价分析，然后调整和优化"生成新的短视频并上传，
通过配置不同短视频平台的更新周期，能够自动重复执行这个过程。

## Solution Name:
中医短视频营销平台

## Stakeholders and Roles:
### Key Stakeholders:
- 创始人
- 风险投资人
- 合作的中医理疗机构

### User Roles/People:
- **Viewer**: 观看短视频的用户
- **Admin**: 管理平台运营的管理员

### System Objectives:
- 在2年内达到1百万的短视频数量
- 保持月用户参与率超过70%
- KPIs: 月活跃用户，留存率，用户反馈的正面评价比例

## Use Case List:
```json
[
    {
        "package": "reviewer",
        "name": "Review Videos",
        "filename": "review_videos.md",
        "actor": "REVIEWER",
        "desc": "审查员审核生成的短视频",
        "priority": "P1",
        "include use cases": ["Trigger every cycle & Generate New Videos", "Upload Videos to Platforms"],
        "extended use cases": ["Adjust and Optimize Videos"]
    },
    {
        "package": "reviewer",
        "name": "Watch Videos Feedback Analysis",
        "filename": "watch_videos_feedback_analysis.md",
        "actor": "REVIEWER",
        "desc": "审查员观看各个短视频的反馈分析",
        "priority": "P1",
        "include use cases": ["Fetch Feedback from Platforms"],
        "extended use cases": []
    },
    {
        "package": "reviewer",
        "name": "Adjust and Optimize Videos",
        "filename": "adjust_and_optimize_videos.md",
        "actor": "REVIEWER",
        "desc": "审查员调整优化短视频",
        "priority": "P1",
        "include use cases": ["Analyze Feedback"],
        "extended use cases": []
    },
    {
        "package": "admin",
        "name": "Configure Videos Platforms",
        "filename": "configure_videos_platforms.md",
        "actor": "ADMIN",
        "desc": "管理员配置不同短视频平台的账号、上传地址、上传规则、爬虫地址和信息获取规则",
        "priority": "P0",
        "include use cases": [],
        "extended use cases": []
    },
    {
        "package": "timer",
        "name": "Trigger every cycle & Generate New Videos",
        "filename": "trigger_every_cycle_&_generate_new_videos.md",
        "actor": "TIMER",
        "desc": "系统生成短视频",
        "priority": "P0",
        "include use cases": [],
        "extended use cases": []
    },
    {
        "package": "timer",
        "name": "Upload Videos to Platforms",
        "filename": "upload_videos_to_platforms.md",
        "actor": "TIMER",
        "desc": "系统将新的短视频上传到多个短视频平台的多个账号",
        "priority": "P0",
        "include use cases": [],
        "extended use cases": []
    },
    {
        "package": "timer",
        "name": "Fetch Feedback from Platforms",
        "filename": "fetch_feedback_from_platforms.md",
        "actor": "TIMER",
        "desc": "系统从短视频平台抓取用户的反馈",
        "priority": "P0",
        "include use cases": [],
        "extended use cases": []
    },
    {
        "package": "timer",
        "name": "Analyze Feedback",
        "filename": "analyze_feedback.md",
        "actor": "TIMER",
        "desc": "系统对用户反馈进行正面和负面评价的分析",
        "priority": "P0",
        "include use cases": [],
        "extended use cases": []
    },
    {
        "package": "timer",
        "name": "Auto Adjust and Optimize Videos",
        "filename": "auto_adjust_and_optimize_videos.md",
        "actor": "TIMER",
        "desc": "系统根据反馈分析结果调整和优化新的短视频",
        "priority": "P0",
        "include use cases": ["Adjust and Optimize Videos"],
        "extended use cases": []
    }
]

```
## Non-functional requirements

### Usability
- 界面设计应该简洁、现代并反映中医的特色。
- 平台需要为中医专家和普通用户提供直观的操作指南。
- 应支持简体中文和繁体中文，确保覆盖更广泛的中医爱好者。

### Reliability
- 系统的正常运行时间应达到99.9%。
- 上传和播放短视频的成功率应超过98%。
- 数据备份机制需要每日进行，确保数据的完整性和安全性。

### Performance
- 视频加载时间不得超过3秒。
- 平台的搜索响应时间应在1秒内。
- 对于高流量请求，系统应能够在5秒内处理并响应。

### Supportability
- 平台需要提供全天候的在线技术支持。
- 平台的所有功能都应支持智能手机、平板和桌面设备。
- 代码结构应简洁，有助于未来的扩展和维护。

### The “+” of the FURPS+

- Design constraints (设计约束)
  - 必须使用关系数据库存储用户数据和视频元数据。
  - 平台界面必须在设计上融合中医的元素。

- Implementation constraints (实现约束)
  - 代码需遵循MVC架构。
  - 所有代码必须有注释，使用Git进行版本控制。

- Interface constraints (接口约束)
  - 平台需能与主流社交媒体平台如微信、微博进行接口对接。
  - 支持常见的支付方式如支付宝、微信支付。

- Physical constraints (物理约束)
  - 托管服务器必须位于中国境内，以确保快速的数据传输速度。

## Anything UNCLEAR:
对于硬件需求需要进行更深入的分析。