"""
SLA（服务水平协议）管理API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.core.config import settings
from app.schemas.admin import SLARule, SLAMetric, SLAViolation
from app.models.admin import SLARule as SLARuleModel
from app.services.sla_enforcement import SLAEnforcementService

logger = logging.getLogger(__name__)

router = APIRouter()

# SLA服务实例
sla_service = SLAEnforcementService()


@router.get("/rules", response_model=List[SLARule])
async def get_sla_rules(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取SLA规则列表"""
    try:
        rules = sla_service.get_rules(db, skip=skip, limit=limit)
        return rules
    except Exception as e:
        logger.error(f"Failed to get SLA rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA rules"
        )


@router.post("/rules", response_model=SLARule)
async def create_sla_rule(
    rule_data: SLARule,
    db: Session = Depends(get_db)
):
    """创建SLA规则"""
    try:
        rule = sla_service.create_rule(db, rule_data)
        return rule
    except Exception as e:
        logger.error(f"Failed to create SLA rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create SLA rule"
        )


@router.get("/rules/{rule_id}", response_model=SLARule)
async def get_sla_rule(
    rule_id: int,
    db: Session = Depends(get_db)
):
    """获取特定SLA规则"""
    try:
        rule = sla_service.get_rule(db, rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SLA rule not found"
            )
        return rule
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SLA rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA rule"
        )


@router.put("/rules/{rule_id}", response_model=SLARule)
async def update_sla_rule(
    rule_id: int,
    rule_data: SLARule,
    db: Session = Depends(get_db)
):
    """更新SLA规则"""
    try:
        rule = sla_service.update_rule(db, rule_id, rule_data)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SLA rule not found"
            )
        return rule
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update SLA rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update SLA rule"
        )


@router.delete("/rules/{rule_id}")
async def delete_sla_rule(
    rule_id: int,
    db: Session = Depends(get_db)
):
    """删除SLA规则"""
    try:
        success = sla_service.delete_rule(db, rule_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SLA rule not found"
            )
        return {"message": "SLA rule deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SLA rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete SLA rule"
        )


@router.get("/metrics", response_model=List[SLAMetric])
async def get_sla_metrics(
    metric_type: Optional[str] = None,
    time_range: Optional[str] = "24h",
    db: Session = Depends(get_db)
):
    """获取SLA指标"""
    try:
        # 解析时间范围
        time_mapping = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }

        time_delta = time_mapping.get(time_range, timedelta(hours=24))
        start_time = datetime.utcnow() - time_delta

        metrics = sla_service.get_metrics(
            db,
            metric_type=metric_type,
            start_time=start_time
        )
        return metrics
    except Exception as e:
        logger.error(f"Failed to get SLA metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA metrics"
        )


@router.get("/violations", response_model=List[SLAViolation])
async def get_sla_violations(
    skip: int = 0,
    limit: int = 100,
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """获取SLA违规记录"""
    try:
        violations = sla_service.get_violations(
            db,
            skip=skip,
            limit=limit,
            severity=severity
        )
        return violations
    except Exception as e:
        logger.error(f"Failed to get SLA violations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA violations"
        )


@router.get("/dashboard")
async def get_sla_dashboard(db: Session = Depends(get_db)):
    """获取SLA仪表板数据"""
    try:
        dashboard_data = sla_service.get_dashboard_data(db)
        return dashboard_data
    except Exception as e:
        logger.error(f"Failed to get SLA dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA dashboard data"
        )


@router.post("/enforce")
async def enforce_sla_rules(
    db: Session = Depends(get_db)
):
    """手动执行SLA规则检查"""
    try:
        result = await sla_service.enforce_rules(db)
        return {
            "message": "SLA enforcement completed",
            "checked_rules": result.get("checked_rules", 0),
            "violations_found": result.get("violations_found", 0),
            "actions_taken": result.get("actions_taken", 0)
        }
    except Exception as e:
        logger.error(f"Failed to enforce SLA rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enforce SLA rules"
        )


@router.get("/compliance")
async def get_sla_compliance(
    time_range: Optional[str] = "7d",
    db: Session = Depends(get_db)
):
    """获取SLA合规性报告"""
    try:
        time_mapping = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }

        time_delta = time_mapping.get(time_range, timedelta(days=7))
        start_time = datetime.utcnow() - time_delta

        compliance = sla_service.get_compliance_report(db, start_time=start_time)
        return compliance
    except Exception as e:
        logger.error(f"Failed to get SLA compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve SLA compliance report"
        )