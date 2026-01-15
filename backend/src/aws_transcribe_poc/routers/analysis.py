# Copyright 2025 Booz Allen Hamilton.
#
# Booz Allen Hamilton Confidential Information.
#
# The contents of this file are the intellectual property of
# Booz Allen Hamilton, Inc. ("BAH") and are subject to copyright protection
# under the laws of the United States and other countries.
#
# You acknowledge that misappropriation, misuse, or redistribution of content
# on the file could cause irreparable harm to BAH and/or to third parties.
#
# You may not copy, reproduce, distribute, publish, display, execute, modify,
# create derivative works of, transmit, sell or offer for resale, or in any way
# exploit any part of this code or program without BAH's express written permission.
#
# The contents of this code or program contains code
# that is itself or was created using artificial intelligence.
#
# To the best of our knowledge, this code does not infringe third-party intellectual
# property rights, contain errors, inaccuracies, bias, or security concerns.
#
# However, Booz Allen does not warrant, claim, or provide any implied
# or express warranty for the aforementioned, nor of merchantability
# or fitness for purpose.
#
# Booz Allen expressly limits liability, whether by contract, tort or in equity
# for any damage or harm caused by use of this artificial intelligence code or program.
#
# Booz Allen is providing this code or program "as is" with the understanding
# that any separately negotiated standards of performance for said code
# or program will be met for the duration of any applicable contract under which
# the code or program is provided.
"""Analysis router for meeting transcript analysis and insights generation.

This module provides endpoints for analyzing meeting transcripts and generating
comprehensive insights including summaries, action items, and compliance checks.
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from ..services.analyzer import AnalyzerService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalyzeRequestModel(BaseModel):
    """Request model for meeting analysis"""

    input_file_path: str
    save_report: bool = False


class AnalyzeResponseModel(BaseModel):
    """Response model for meeting analysis"""

    success: bool
    final_report: dict | None = None
    output_file_path: str | None = None
    error: str | None = None


@router.post("/analyze", response_model=AnalyzeResponseModel)
async def analyze_meeting(request: AnalyzeRequestModel) -> AnalyzeResponseModel:
    """Analyze a meeting transcript and generate insights.

    Takes in a JSON file containing transcript and user notes and generates
    a comprehensive analysis including:
    - Summary
    - Action Items
    - Inconsistencies (optional)
    - Compliance issues (optional)
    - Meeting improvements (optional)

    Args:
        request: Request containing:
            - input_file_path: Path to JSON file with meeting context
            - save_report: whether to save report to output/ folder (default: False)

    Returns:
        AnalyzeResponseModel with analysis results and optional file path (if save_report=True)
    """
    try:
        input_path = Path(request.input_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {request.input_file_path}")

        analyzer_service = AnalyzerService(request.input_file_path)

        report, output_path = analyzer_service.run_analysis(
            save_report=request.save_report
        )

        return AnalyzeResponseModel(
            success=True, final_report=report.model_dump(), output_file_path=output_path
        )
    except FileNotFoundError as e:
        return AnalyzeResponseModel(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return AnalyzeResponseModel(success=False, error=str(e))
