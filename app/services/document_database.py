#!/usr/bin/env python3
"""
基于券商研报创建模拟文档数据库
用于RAG系统检索测试
"""

import json
import os
from typing import List, Dict, Optional
import random
from pathlib import Path

class DocumentDatabase:
    def __init__(self):
        self.documents = []
        self.load_sample_documents()

    def load_sample_documents(self):
        """加载示例文档（基于真实券商研报内容模拟）"""

        # 基于研报文件名和内容创建模拟文档
        sample_docs = [
            {
                "id": "doc_001",
                "title": "比亚迪新能源汽车业务分析报告",
                "content": """比亚迪作为中国领先的新能源汽车制造商，2023年在新能源汽车领域表现突出。
                公司主要业务包括新能源汽车、动力电池、半导体等。在新能源汽车市场，比亚迪凭借刀片电池技术
                和DM-i混动技术占据了市场领先地位。2023年比亚迪新能源汽车销量超过180万辆，同比增长超过70%。
                公司在电池技术方面持续创新，刀片电池具有高安全性、长寿命等特点。在智能化方面，比亚迪推出了
                DiLink智能网联系统和DiPilot智能驾驶辅助系统。公司还积极拓展海外市场，在欧洲、东南亚等地区
                建立了生产基地和销售网络。分析师认为比亚迪在新能源汽车市场的竞争优势明显，未来增长潜力较大。
                风险因素包括市场竞争加剧、原材料价格波动、政策变化等。""",
                "metadata": {
                    "source": "券商研报",
                    "company": "比亚迪",
                    "industry": "新能源汽车",
                    "date": "2023-12-01",
                    "author": "中信证券",
                    "file_type": "pdf",
                    "pages": 25
                },
                "keywords": ["比亚迪", "新能源汽车", "动力电池", "刀片电池", "DM-i混动", "销量增长", "市场领先", "海外拓展"]
            },
            {
                "id": "doc_002",
                "title": "半导体行业投资机会分析",
                "content": """半导体行业作为数字经济的基础设施，在人工智能、5G、物联网等新兴技术推动下
                迎来新一轮增长周期。2023年全球半导体市场规模达到5700亿美元，预计2024年将增长12%。
                中国半导体产业在政策支持下快速发展，国产化率持续提升。芯片设计、制造、封装测试等环节
                都有重要突破。在AI芯片领域，国内企业如寒武纪、海光信息等推出了具有竞争力的产品。
                存储芯片方面，长江存储、长鑫存储等企业技术水平不断提升。功率半导体和模拟芯片领域
                也有重要进展。分析师认为，半导体行业长期投资价值显著，但需要关注技术迭代风险、
                地缘政治影响、市场周期波动等因素。建议重点关注具有核心技术优势和客户资源的企业。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "半导体",
                    "company": "多家",
                    "date": "2023-11-15",
                    "author": "海通证券",
                    "file_type": "pdf",
                    "pages": 32
                },
                "keywords": ["半导体", "AI芯片", "存储芯片", "功率半导体", "国产化", "市场规模", "投资价值", "技术迭代"]
            },
            {
                "id": "doc_003",
                "title": "人工智能行业深度研究报告",
                "content": """人工智能技术正在经历快速发展，大语言模型成为2023年的热点。
                ChatGPT、GPT-4等大模型展示了强大的自然语言处理能力，推动了AIGC（AI生成内容）
                应用的爆发式增长。国内AI企业如百度、阿里、腾讯、字节跳动等都在积极布局大模型。
                百度推出了文心一言，阿里推出了通义千问，腾讯推出了混元大模型。这些模型在中文
                处理方面具有优势。AI应用场景不断扩大，包括智能客服、内容生成、代码辅助、
                教育培训等。分析师认为AI产业链的投资机会主要在算力基础设施、大模型开发、
                垂直应用三个层面。需要关注技术发展、监管政策、商业化落地等因素。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "人工智能",
                    "company": "多家",
                    "date": "2023-12-10",
                    "author": "华泰证券",
                    "file_type": "pdf",
                    "pages": 45
                },
                "keywords": ["人工智能", "大语言模型", "ChatGPT", "GPT-4", "AIGC", "文心一言", "通义千问", "投资机会"]
            },
            {
                "id": "doc_004",
                "title": "中信证券投资价值分析",
                "content": """中信证券作为中国领先的综合性证券公司，业务涵盖证券经纪、投资银行、
                资产管理、研究咨询等多个领域。2023年公司业绩表现稳健，营业收入和净利润保持增长。
                在经纪业务方面，公司拥有广泛的客户基础和强大的销售网络。投资银行业务在IPO、
                再融资、并购重组等方面保持领先地位。资产管理业务规模持续扩大，产品线日益丰富。
                研究团队实力雄厚，覆盖多个行业和领域。公司积极布局金融科技，提升数字化服务能力。
                风险控制体系完善，合规管理水平较高。分析师认为，中信证券在行业竞争中具有明显优势，
                长期投资价值突出。需要关注市场波动、监管政策变化、行业竞争等因素。""",
                "metadata": {
                    "source": "券商研报",
                    "company": "中信证券",
                    "industry": "证券",
                    "date": "2023-10-20",
                    "author": "国泰君安",
                    "file_type": "pdf",
                    "pages": 28
                },
                "keywords": ["中信证券", "证券经纪", "投资银行", "资产管理", "研究咨询", "金融科技", "投资价值", "风险控制"]
            },
            {
                "id": "doc_005",
                "title": "中国汽车市场新变局分析",
                "content": """中国汽车市场正在经历深刻变革，新能源汽车快速崛起，传统燃油车市场份额
                持续下降。2023年新能源汽车渗透率超过30%，预计2025年将达到50%。市场竞争格局发生重大变化，
                比亚迪、特斯拉等新能源车企领先优势明显，传统车企如长城汽车、吉利汽车等正在加快转型。
                智能化成为汽车发展的重要方向，自动驾驶、智能座舱等技术快速发展。汽车产业链正在重构，
                电池、芯片、软件等关键环节的重要性不断提升。中国品牌在全球市场的影响力逐步增强。
                分析师认为，汽车行业的投资机会主要集中在新能源汽车、智能化技术、核心零部件等领域。
                需要关注技术发展、政策支持、市场需求等因素。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "汽车",
                    "company": "多家",
                    "date": "2023-09-15",
                    "author": "招商证券",
                    "file_type": "pdf",
                    "pages": 35
                },
                "keywords": ["中国汽车市场", "新能源汽车", "比亚迪", "特斯拉", "长城汽车", "智能化", "自动驾驶", "汽车产业链"]
            },
            {
                "id": "doc_006",
                "title": "高端白酒市场分析报告",
                "content": """中国高端白酒市场呈现稳健发展态势，消费升级推动行业增长。
                茅台、五粮液、剑南春等高端品牌保持领先地位。2023年高端白酒市场规模超过2000亿元，
                预计未来5年复合增长率约8%。消费群体逐步年轻化，80后、90后成为重要消费力量。
                品牌价值持续提升，茅台作为中国高端白酒的代表，品牌价值超过3000亿元。
                渠道体系不断完善，线上线下融合发展趋势明显。分析师认为，高端白酒行业具有
                稳健的投资价值，重点关注品牌力强、渠道优势明显的企业。风险因素包括消费环境变化、
                行业竞争加剧、政策监管等。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "白酒",
                    "company": "茅台、五粮液等",
                    "date": "2023-11-05",
                    "author": "安信证券",
                    "file_type": "pdf",
                    "pages": 22
                },
                "keywords": ["高端白酒", "茅台", "五粮液", "消费升级", "品牌价值", "渠道体系", "投资价值", "消费群体"]
            },
            {
                "id": "doc_007",
                "title": "ChatGPT在A股的投资机会",
                "content": """ChatGPT作为突破性的AI应用，为A股市场带来新的投资机会。
                算力需求激增推动GPU、服务器等硬件厂商受益。大模型训练需要大量算力资源，
                相关服务器、芯片、IDC等企业有望获得增长。AIGC应用场景丰富，内容创作、
                教育培训、智能客服等领域将率先落地。A股AI相关概念股表现活跃，
                市场关注度高。分析师认为，ChatGPT产业链的投资机会主要在算力、算法、
                数据、应用四个层面。需要关注技术发展、商业化进展、监管政策等因素。
                短期概念炒作风险较大，长期看好具有核心技术优势的企业。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "计算机",
                    "company": "多家",
                    "date": "2023-02-05",
                    "author": "东北证券",
                    "file_type": "pdf",
                    "pages": 18
                },
                "keywords": ["ChatGPT", "A股投资", "算力需求", "GPU", "AIGC", "大模型", "投资机会", "概念股"]
            },
            {
                "id": "doc_008",
                "title": "钠离子电池行业前景分析",
                "content": """钠离子电池作为新兴的储能技术，具有成本低、资源丰富、安全性好等优势。
                相比锂电池，钠资源储量丰富且分布均匀，成本优势明显。技术方面，钠离子电池
                能量密度已达到160Wh/kg，循环寿命超过3000次，基本满足储能应用需求。
                宁德时代、比亚迪、中科海钠等企业在钠离子电池领域积极布局。2023年钠离子电池
                开始产业化，预计2025年市场规模将达到100亿元。应用场景包括储能系统、
        低速电动车、两轮车等。分析师认为，钠离子电池在储能领域具有巨大潜力，
        相关企业投资价值突出。需要关注技术进步、成本下降、市场需求等因素。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "新能源",
                    "company": "宁德时代、比亚迪等",
                    "date": "2023-03-27",
                    "author": "华金证券",
                    "file_type": "pdf",
                    "pages": 24
                },
                "keywords": ["钠离子电池", "储能技术", "宁德时代", "比亚迪", "成本优势", "产业化", "投资价值", "应用场景"]
            },
            {
                "id": "doc_009",
                "title": "华为AI盘古大模型研究",
                "content": """华为推出的盘古大模型系列包括NLP大模型、CV大模型、科学计算大模型等。
                盘古NLP大模型参数规模超过1000亿，在中文理解方面表现优异。CV大模型在
                图像识别、目标检测等任务上达到业界领先水平。科学计算大模型应用于
                天气预报、药物研发等领域。华为依托昇腾AI芯片和MindSpore框架，
                构建了完整的AI生态体系。盘古大模型已在金融、制造、医疗等行业
                得到应用。分析师认为，华为在AI领域具有全栈技术优势，大模型
                商业化前景广阔。需要关注技术迭代、市场竞争、政策环境等因素。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "人工智能",
                    "company": "华为",
                    "date": "2023-03-25",
                    "author": "浙商证券",
                    "file_type": "pdf",
                    "pages": 30
                },
                "keywords": ["华为", "盘古大模型", "NLP大模型", "昇腾芯片", "MindSpore", "AI生态", "商业化", "全栈技术"]
            },
            {
                "id": "doc_010",
                "title": "GPU行业深度报告",
                "content": """GPU作为AI计算的核心硬件，在人工智能时代发挥重要作用。
                英伟达凭借CUDA生态和技术优势占据市场领先地位，市场份额超过80%。
                AMD、英特尔等厂商也在积极追赶。中国GPU产业在政策支持下快速发展，
                景嘉微、壁仞科技、摩尔线程等企业推出了具有竞争力的产品。
                AI训练和推理对GPU性能要求不断提升，推动了GPU技术的快速发展。
                市场规模方面，2023年全球GPU市场超过500亿美元，预计2025年将
                达到800亿美元。分析师认为，GPU行业具有长期增长潜力，重点关注
                具有核心技术优势的企业。风险因素包括技术迭代、市场竞争、
                地缘政治等。""",
                "metadata": {
                    "source": "券商研报",
                    "industry": "半导体",
                    "company": "英伟达、AMD、英特尔等",
                    "date": "2023-03-26",
                    "author": "华金证券",
                    "file_type": "pdf",
                    "pages": 40
                },
                "keywords": ["GPU", "AI计算", "英伟达", "CUDA生态", "景嘉微", "市场规模", "技术迭代", "投资机会"]
            }
        ]

        self.documents = sample_docs

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """基于关键词匹配的简单检索"""
        query_lower = query.lower()
        scored_docs = []

        for doc in self.documents:
            score = 0

            # 标题匹配
            if query_lower in doc['title'].lower():
                score += 10

            # 关键词匹配
            for keyword in doc.get('keywords', []):
                if query_lower in keyword.lower():
                    score += 5

            # 内容匹配
            query_words = query_lower.split()
            content_words = doc['content'].lower().split()
            matches = sum(1 for word in query_words if word in content_words)
            score += matches

            if score > 0:
                scored_docs.append({
                    'document': doc,
                    'score': score
                })

        # 按分数排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)

        # 返回top_k结果
        return [item['document'] for item in scored_docs[:top_k]]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """根据ID获取文档"""
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def save_database(self, filepath: str):
        """保存文档数据库"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load_database(self, filepath: str):
        """加载文档数据库"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

# 创建并保存数据库
if __name__ == "__main__":
    db = DocumentDatabase()
    db.save_database("document_database.json")
    print(f"文档数据库已创建，包含 {len(db.documents)} 个文档")