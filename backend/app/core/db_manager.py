#!/usr/bin/env python3
"""
数据库管理CLI工具
使用SQLAlchemy代替Alembic
"""

import sys
import os
import argparse

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.core.db_init import db_initializer
from app.core.config import settings


def init_db():
    """初始化数据库"""
    print("🔧 初始化数据库...")
    try:
        success = db_initializer.init_database()
        if success:
            print("✅ 数据库初始化成功")
        else:
            print("❌ 数据库初始化失败")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 数据库初始化错误: {e}")
        sys.exit(1)


def check_status():
    """检查数据库状态"""
    print("📊 检查数据库状态...")
    try:
        status = db_initializer.check_database_status()

        print(f"状态: {status['status']}")

        if status['status'] == 'healthy':
            print("📋 数据库表:")
            for table in status['tables']:
                count = status['counts'].get(table, 'unknown')
                print(f"  - {table}: {count} 条记录")
        elif status['status'] == 'error':
            print(f"❌ 错误: {status.get('error', '未知错误')}")
        else:
            print("⚠️ 数据库为空")

    except Exception as e:
        print(f"❌ 检查状态失败: {e}")
        sys.exit(1)


def reset_db():
    """重置数据库"""
    print("⚠️ 重置数据库将删除所有数据，确认操作吗？")
    confirm = input("输入 'yes' 确认: ")

    if confirm.lower() != 'yes':
        print("❌ 操作已取消")
        return

    print("🔄 重置数据库...")
    try:
        success = db_initializer.reset_database()
        if success:
            print("✅ 数据库重置成功")
        else:
            print("❌ 数据库重置失败")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 数据库重置错误: {e}")
        sys.exit(1)


def create_tables():
    """仅创建表结构"""
    print("🏗️ 创建数据库表...")
    try:
        db_initializer.create_tables()
        print("✅ 数据库表创建成功")
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        sys.exit(1)


def insert_data():
    """插入初始数据"""
    print("📝 插入初始数据...")
    try:
        from app.core.database import SessionLocal
        with SessionLocal() as session:
            db_initializer._insert_system_configs(session)
            session.commit()
        print("✅ 初始数据插入成功")
    except Exception as e:
        print(f"❌ 插入数据失败: {e}")
        sys.exit(1)


def create_indexes():
    """创建索引"""
    print("🔍 创建数据库索引...")
    try:
        db_initializer.create_indexes()
        print("✅ 索引创建成功")
    except Exception as e:
        print(f"❌ 创建索引失败: {e}")
        sys.exit(1)


def show_info():
    """显示数据库信息"""
    print("📋 数据库配置信息:")
    print(f"  - 数据库URL: {settings.database_url}")
    print(f"  - Redis URL: {settings.redis_url}")
    print(f"  - Milvus: {settings.milvus_host}:{settings.milvus_port}")
    print(f"  - Neo4j: {settings.neo4j_uri}")
    print(f"  - MinIO: {settings.minio_endpoint}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据库管理工具')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 子命令
    subparsers.add_parser('init', help='初始化数据库')
    subparsers.add_parser('status', help='检查数据库状态')
    subparsers.add_parser('reset', help='重置数据库')
    subparsers.add_parser('create-tables', help='创建表结构')
    subparsers.add_parser('insert-data', help='插入初始数据')
    subparsers.add_parser('create-indexes', help='创建索引')
    subparsers.add_parser('info', help='显示配置信息')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 执行对应命令
    if args.command == 'init':
        init_db()
    elif args.command == 'status':
        check_status()
    elif args.command == 'reset':
        reset_db()
    elif args.command == 'create-tables':
        create_tables()
    elif args.command == 'insert-data':
        insert_data()
    elif args.command == 'create-indexes':
        create_indexes()
    elif args.command == 'info':
        show_info()
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()