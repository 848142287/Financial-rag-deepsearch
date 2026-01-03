#!/usr/bin/env python3
"""
券商研报评测数据集生成器
从券商研报PDF文档中提炼500个不同复杂度的问题用于评测系统
"""

import re
import json
import random
from pathlib import Path
from typing import List, Dict
from app.core.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dataset_generator')

class DatasetGenerator:
    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)
        self.questions = []
        self.companies = set()
        self.industries = set()
        self.topics = set()

    def extract_text_from_filename(self, filepath: Path) -> str:
        """从文件名提取关键信息"""
        filename = filepath.stem
        # 移除日期前缀和常见无用信息
        clean_name = re.sub(r'^\d{8}-?', '', filename)
        clean_name = re.sub(r'^\d{4}-\d{2}-\d{2}-?', '', filename)
        return clean_name

    def analyze_reports_structure(self) -> Dict[str, List[str]]:
        """分析研报结构，提取公司、行业、主题等信息"""
        structure = {
            'companies': set(),
            'industries': set(),
            'topics': set(),
            'file_categories': set()
        }

        pdf_files = list(self.reports_dir.rglob("*.pdf"))
        sample_files = random.sample(pdf_files, min(100, len(pdf_files)))  # 采样100个文件分析

        for filepath in sample_files:
            filename = filepath.name
            clean_name = self.extract_text_from_filename(filepath)

            # 提取公司名称
            company_patterns = [
                r'(中信证券|海通证券|国泰君安|华泰证券|招商证券|广发证券|中金公司|申万宏源|银河证券|国信证券|东方证券|兴业证券|安信证券|东北证券|浙商证券|华金证券|德勤|毕马威)',
                r'(比亚迪|长城汽车|蔚来汽车|小鹏汽车|理想汽车|特斯拉|腾讯|阿里巴巴|百度|京东|美团|小米|华为|苹果|微软|谷歌|亚马逊|英伟达|英特尔|AMD|台积电|三星)',
                r'(茅台|五粮液|剑南春|泸州老窖|古井贡酒|汾酒|洋河股份|口子窖|水井坊)',
                r'(中国平安|中国人寿|中国人保|新华保险|中国太保|友邦保险)',
                r'(工商银行|建设银行|农业银行|中国银行|交通银行|招商银行|平安银行|浦发银行|民生银行|兴业银行|光大银行)',
                r'(中国石油|中国石化|中海油|中国神华|陕西煤业|兖州煤业|潞安环能|山西焦煤)',
                r'(宝钢股份|河钢股份|沙钢股份|鞍钢股份|首钢股份)',
                r'(万科|保利地产|中国恒大|碧桂园|融创中国|华夏幸福)',
            ]

            for pattern in company_patterns:
                matches = re.findall(pattern, filename)
                structure['companies'].update(matches)

            # 提取行业
            industry_patterns = [
                r'(新能源汽车|汽车|整车|零部件)', r'(半导体|芯片|集成电路|EDA)', r'(人工智能|AI|算力|大模型|ChatGPT)',
                r'(医药|生物|疫苗|医疗器械)', r'(消费|零售|白酒|食品)', r'(银行|保险|证券|金融)',
                r'(房地产|建筑|建材)', r'(化工|石油|煤炭|有色|钢铁)', r'(电力|新能源|光伏|风电)',
                r'(计算机|软件|互联网|游戏)', r'(机械|装备|军工)', r'(农林牧渔|农业)',
                r'(通信|5G|云计算|大数据)', r'(环保|公用事业)', r'(交运|物流|航空|航运)'
            ]

            for pattern in industry_patterns:
                matches = re.findall(pattern, filename)
                structure['industries'].update(matches)

            # 提取主题
            topic_patterns = [
                r'(研报|报告|分析|研究|调研)', r'(投资|策略|配置|组合)', r'(财报|业绩|盈利)',
                r'(技术|研发|创新|突破)', r'(市场|行业|产业)', r'(政策|法规|监管)',
                r'(并购|重组|整合)', r'(估值|评级|目标价)', r'(风险|挑战|机遇)',
                r'(趋势|展望|预测)', r'(深度|专题|框架)', r'(产业链|供应链)',
                r'(竞争|格局|份额)', r'(成本|价格|利润)', r'(增长|扩张|放缓)'
            ]

            for pattern in topic_patterns:
                matches = re.findall(pattern, filename)
                structure['topics'].update(matches)

            # 分类
            if '晨会' in filename:
                structure['file_categories'].add('晨会报告')
            elif '深度' in filename or '专题' in filename:
                structure['file_categories'].add('深度报告')
            elif '策略' in filename:
                structure['file_categories'].add('策略报告')
            elif '行业' in filename:
                structure['file_categories'].add('行业报告')
            elif '公司' in filename:
                structure['file_categories'].add('公司报告')
            elif '晨会' in filename:
                structure['file_categories'].add('晨会纪要')

        # 转换set为list并去重
        for key in structure:
            structure[key] = list(structure[key])[:50]  # 限制每类最多50个

        return structure

    def generate_simple_questions(self, structure: Dict[str, List[str]], count: int = 150) -> List[Dict]:
        """生成简单复杂度的问题（基础信息查询）"""
        questions = []

        # 基础事实查询
        templates = [
            "{company}的主营业务是什么？",
            "{company}的最新市值是多少？",
            "{industry}行业的市场规模有多大？",
            "{company}在{industry}行业中的地位如何？",
            "{company}的主要竞争对手有哪些？",
            "{industry}行业的发展趋势是什么？",
            "{company}的财务表现如何？",
            "{industry}行业的政策环境如何？",
            "{company}的核心竞争优势是什么？",
            "{topic}对{industry}行业有什么影响？"
        ]

        for i in range(count):
            template = random.choice(templates)
            question = template.format(
                company=random.choice(structure['companies']) if '{company}' in template else '',
                industry=random.choice(structure['industries']) if '{industry}' in template else '',
                topic=random.choice(structure['topics']) if '{topic}' in template else ''
            )

            questions.append({
                'id': f'q_simple_{i+1:03d}',
                'question': question,
                'complexity': 'simple',
                'category': '基础信息查询',
                'difficulty_score': random.randint(1, 3),
                'expected_answer_type': '事实陈述',
                'keywords': [q for q in question.split() if len(q) > 2][:3]
            })

        return questions

    def generate_medium_questions(self, structure: Dict[str, List[str]], count: int = 200) -> List[Dict]:
        """生成中等复杂度的问题（分析比较）"""
        questions = []

        templates = [
            "比较{company1}和{company2}在{industry}领域的竞争策略",
            "分析{industry}行业的{topic}对投资决策的影响",
            "评估{company}在{topic}方面的风险和机遇",
            "对比{industry1}和{industry2}行业的发展前景",
            "分析{topic}对{company}业务模式的冲击",
            "评估{industry}行业政策变化对投资组合的影响",
            "比较{company}在{topic1}和{topic2}方面的表现",
            "分析{industry}行业{topic}的市场机会",
            "评估{company}相对于竞争对手的技术优势",
            "对比{company1}和{company2}的财务健康状况"
        ]

        for i in range(count):
            template = random.choice(templates)
            question = template.format(
                company1=random.choice(structure['companies']) if '{company1}' in template else '',
                company2=random.choice(structure['companies']) if '{company2}' in template else '',
                company=random.choice(structure['companies']) if '{company}' in template else '',
                industry=random.choice(structure['industries']) if '{industry}' in template else '',
                industry1=random.choice(structure['industries']) if '{industry1}' in template else '',
                industry2=random.choice(structure['industries']) if '{industry2}' in template else '',
                topic=random.choice(structure['topics']) if '{topic}' in template else '',
                topic1=random.choice(structure['topics']) if '{topic1}' in template else '',
                topic2=random.choice(structure['topics']) if '{topic2}' in template else ''
            )

            questions.append({
                'id': f'q_medium_{i+1:03d}',
                'question': question,
                'complexity': 'medium',
                'category': '分析比较',
                'difficulty_score': random.randint(4, 7),
                'expected_answer_type': '综合分析',
                'keywords': [q for q in question.split() if len(q) > 2][:4]
            })

        return questions

    def generate_complex_questions(self, structure: Dict[str, List[str]], count: int = 150) -> List[Dict]:
        """生成高复杂度的问题（深度分析预测）"""
        questions = []

        templates = [
            "综合分析{company1}、{company2}、{company3}在{industry}领域的战略布局和未来3年发展趋势",
            "深度评估{topic}对{industry1}和{industry2}行业的长期影响及投资机会",
            "基于当前{topic}趋势，预测{company}未来5年的业务发展路径和投资价值",
            "综合分析{industry}行业政策变化、技术发展和市场竞争对{company}的多维度影响",
            "评估{company}在{topic1}、{topic2}、{topic3}等多个领域的综合竞争优势",
            "深度分析{industry}产业链上下游变化对{company}供应链管理的战略建议",
            "综合评估{topic}对{industry}行业投资组合的重构建议和风险管理策略",
            "基于{topic1}和{topic2}的交叉影响，预测{industry}行业的发展拐点和投资时机",
            "深度分析{company}在全球{industry}竞争格局中的战略定位和突破路径",
            "综合评估{topic}、{industry}政策和{company}基本面三重因素的投资决策框架"
        ]

        for i in range(count):
            template = random.choice(templates)
            question = template.format(
                company1=random.choice(structure['companies']) if '{company1}' in template else '',
                company2=random.choice(structure['companies']) if '{company2}' in template else '',
                company3=random.choice(structure['companies']) if '{company3}' in template else '',
                company=random.choice(structure['companies']) if '{company}' in template else '',
                industry=random.choice(structure['industries']) if '{industry}' in template else '',
                industry1=random.choice(structure['industries']) if '{industry1}' in template else '',
                industry2=random.choice(structure['industries']) if '{industry2}' in template else '',
                topic=random.choice(structure['topics']) if '{topic}' in template else '',
                topic1=random.choice(structure['topics']) if '{topic1}' in template else '',
                topic2=random.choice(structure['topics']) if '{topic2}' in template else '',
                topic3=random.choice(structure['topics']) if '{topic3}' in template else ''
            )

            questions.append({
                'id': f'q_complex_{i+1:03d}',
                'question': question,
                'complexity': 'complex',
                'category': '深度分析预测',
                'difficulty_score': random.randint(8, 10),
                'expected_answer_type': '综合评估预测',
                'keywords': [q for q in question.split() if len(q) > 2][:5]
            })

        return questions

    def generate_dataset(self, total_count: int = 500) -> List[Dict]:
        """生成完整的评测数据集"""
        logger.info(f"开始从 {self.reports_dir} 生成评测数据集...")

        # 分析研报结构
        logger.info("分析研报文档结构...")
        structure = self.analyze_reports_structure()

        logger.info(f"发现公司: {len(structure['companies'])}个")
        logger.info(f"发现行业: {len(structure['industries'])}个")
        logger.info(f"发现主题: {len(structure['topics'])}个")

        # 生成不同复杂度的问题
        simple_count = int(total_count * 0.3)  # 30% 简单问题
        medium_count = int(total_count * 0.4)  # 40% 中等问题
        complex_count = total_count - simple_count - medium_count  # 30% 复杂问题

        logger.info(f"生成问题: 简单 {simple_count}个, 中等 {medium_count}个, 复杂 {complex_count}个")

        # 生成各类问题
        simple_questions = self.generate_simple_questions(structure, simple_count)
        medium_questions = self.generate_medium_questions(structure, medium_count)
        complex_questions = self.generate_complex_questions(structure, complex_count)

        # 合并所有问题
        all_questions = simple_questions + medium_questions + complex_questions

        # 打乱顺序
        random.shuffle(all_questions)

        # 重新编号
        for i, q in enumerate(all_questions, 1):
            q['id'] = f'q_{i:03d}'
            q['dataset_order'] = i

        logger.info(f"成功生成 {len(all_questions)} 个评测问题")

        return all_questions, structure

def main():
    # 配置路径
    reports_dir = "/Users/mac/AITestProject/Financial-rag-deepsearch/券商研报"
    output_file = "/Users/mac/AITestProject/Financial-rag-deepsearch/dataset_evaluation.json"

    # 创建生成器
    generator = DatasetGenerator(reports_dir)

    # 生成数据集
    questions, structure = generator.generate_dataset(500)

    # 创建完整的数据集
    dataset = {
        'metadata': {
            'name': '券商研报评测数据集',
            'version': '1.0.0',
            'created_date': '2025-12-17',
            'total_questions': len(questions),
            'source_reports_count': 864,  # 实际PDF文件数
            'source_directory': str(reports_dir)
        },
        'structure_analysis': structure,
        'questions': questions,
        'complexity_distribution': {
            'simple': len([q for q in questions if q['complexity'] == 'simple']),
            'medium': len([q for q in questions if q['complexity'] == 'medium']),
            'complex': len([q for q in questions if q['complexity'] == 'complex'])
        },
        'categories': {
            '基础信息查询': len([q for q in questions if q['category'] == '基础信息查询']),
            '分析比较': len([q for q in questions if q['category'] == '分析比较']),
            '深度分析预测': len([q for q in questions if q['category'] == '深度分析预测'])
        }
    }

    # 保存数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"评测数据集已保存到: {output_file}")

    # 输出统计信息
    print("\n=== 评测数据集统计信息 ===")
    print(f"总问题数: {len(questions)}")
    print(f"复杂度分布:")
    print(f"  - 简单: {dataset['complexity_distribution']['simple']} 个")
    print(f"  - 中等: {dataset['complexity_distribution']['medium']} 个")
    print(f"  - 复杂: {dataset['complexity_distribution']['complex']} 个")
    print(f"分类分布:")
    for category, count in dataset['categories'].items():
        print(f"  - {category}: {count} 个")

    # 示例问题展示
    print("\n=== 示例问题展示 ===")
    simple_examples = [q for q in questions if q['complexity'] == 'simple'][:3]
    medium_examples = [q for q in questions if q['complexity'] == 'medium'][:3]
    complex_examples = [q for q in questions if q['complexity'] == 'complex'][:3]

    print("\n简单问题示例:")
    for q in simple_examples:
        print(f"  {q['id']}: {q['question']}")

    print("\n中等问题示例:")
    for q in medium_examples:
        print(f"  {q['id']}: {q['question']}")

    print("\n复杂问题示例:")
    for q in complex_examples:
        print(f"  {q['id']}: {q['question']}")

    return dataset

if __name__ == "__main__":
    dataset = main()