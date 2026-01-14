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

"""Notes Normalizer Service.

This module provides functionality to normalize meeting notes from text files
into a structured JSON format organized by attendee.

Expected input format:
    Notes can be organized by attendee using brackets, e.g.:

    [Eli Thompson]
    - Migration timeline: 2 weeks
    - Can skip security review

    [Fran Reyes]
    - Timeline conflict
    - Security review concern

    If no author markers are found, all notes are attributed to "unknown".
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_AUTHOR = "unknown"


@dataclass
class AttendeeNotes:
    """Represents notes from a single attendee."""

    name: str
    raw_notes: str


@dataclass
class NormalizedNotes:
    """Normalized notes output structure."""

    attendee_notes: dict[str, AttendeeNotes] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "attendee_notes": {
                name: {"raw_notes": notes.raw_notes}
                for name, notes in self.attendee_notes.items()
            }
        }


class NotesNormalizer:
    """Service to normalize meeting notes text files by attendee."""

    # Pattern to match attendee headers like [Name] or [First Last]
    ATTENDEE_PATTERN = re.compile(r"^\s*\[([^\]]+)\]\s*$", re.MULTILINE)

    def __init__(self):
        pass

    def normalize(self, content: str) -> NormalizedNotes:
        """Normalize meeting notes text into attendee-grouped structure.

        Parses text content looking for attendee markers in brackets (e.g., [Name]).
        Each attendee's notes are collected until the next attendee marker.
        If no markers are found, all content is attributed to "unknown".

        Args:
            content: The raw text content of the notes file.

        Returns:
            NormalizedNotes with notes organized by attendee.
        """
        result = NormalizedNotes()

        # Find all attendee markers and their positions
        matches = list(self.ATTENDEE_PATTERN.finditer(content))

        if not matches:
            # No attendee markers found - attribute all to "unknown"
            cleaned_notes = content.strip()
            if cleaned_notes:
                result.attendee_notes[DEFAULT_AUTHOR] = AttendeeNotes(
                    name=DEFAULT_AUTHOR,
                    raw_notes=cleaned_notes,
                )
            logger.info("No attendee markers found, attributed notes to 'unknown'")
            return result

        # Process each attendee section
        for i, match in enumerate(matches):
            attendee_name = match.group(1).strip()

            # Determine the end of this attendee's section
            section_start = match.end()
            if i + 1 < len(matches):
                section_end = matches[i + 1].start()
            else:
                section_end = len(content)

            # Extract and clean the notes for this attendee
            raw_notes = content[section_start:section_end].strip()

            if attendee_name in result.attendee_notes:
                # Append to existing notes if attendee appears multiple times
                existing = result.attendee_notes[attendee_name].raw_notes
                result.attendee_notes[
                    attendee_name
                ].raw_notes = f"{existing}\n{raw_notes}"
            else:
                result.attendee_notes[attendee_name] = AttendeeNotes(
                    name=attendee_name,
                    raw_notes=raw_notes,
                )

        logger.info(f"Normalized notes: {len(result.attendee_notes)} attendees found")

        return result
