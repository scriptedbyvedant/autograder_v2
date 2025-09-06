# File: database/postgres_handler.py

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

class PostgresHandler:
    def __init__(self, conn_params=None):
        if conn_params is None:
            conn_params = {
                'host':     'localhost',
                'port':     5432,
                'database': 'autograder_db',
                'user':     'vedant',
                'password': 'vedant'
            }
        self.conn_params = conn_params
        self.conn = None
        self.connect()

    def connect(self):
        if self.conn is None or getattr(self.conn, 'closed', True):
            self.conn = psycopg2.connect(**self.conn_params)

    def close(self):
        if self.conn and not getattr(self.conn, 'closed', True):
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # UPSERT method (properly inside the class)
    # ------------------------------------------------------------------
    def insert_or_update_grading_result(
        self,
        student_id: str,
        professor_id: str,
        course: str,
        semester: str,
        assignment_no: str,
        question: str,
        student_answer: str,
        language: str,
        old_score: float,
        new_score: float,
        old_feedback: str,
        new_feedback: str
    ) -> int:
        """
        Upsert a grading result on (student_id, assignment_no, question).
        If a row exists, update; otherwise insert.
        Returns the row id.
        NOTE: Make sure you have a unique index:
          CREATE UNIQUE INDEX IF NOT EXISTS uq_results
          ON grading_results (student_id, assignment_no, question);
        """
        self.connect()
        sql = """
        INSERT INTO grading_results
          (student_id, professor_id, course, semester, assignment_no, question,
           student_answer, language, old_score, new_score, old_feedback, new_feedback, created_at)
        VALUES (%(student_id)s, %(professor_id)s, %(course)s, %(semester)s, %(assignment_no)s, %(question)s,
                %(student_answer)s, %(language)s, %(old_score)s, %(new_score)s, %(old_feedback)s, %(new_feedback)s, %(created_at)s)
        ON CONFLICT (student_id, assignment_no, question) DO UPDATE
          SET new_score      = EXCLUDED.new_score,
              new_feedback   = EXCLUDED.new_feedback,
              student_answer = EXCLUDED.student_answer,
              language       = EXCLUDED.language,
              course         = EXCLUDED.course,
              semester       = EXCLUDED.semester,
              professor_id   = EXCLUDED.professor_id
        RETURNING id;
        """
        params = dict(
            student_id=student_id,
            professor_id=professor_id,
            course=course,
            semester=semester,
            assignment_no=assignment_no,
            question=question,
            student_answer=student_answer,
            language=language,
            old_score=old_score,
            new_score=new_score,
            old_feedback=old_feedback,
            new_feedback=new_feedback,
            created_at=datetime.now()
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rid = cur.fetchone()[0]
        self.conn.commit()
        return rid

    # ------------------------------------------------------------------
    # Backwards-compatible method (delegates to UPSERT)
    # ------------------------------------------------------------------
    def insert_grading_result(self,
                              student_id: str,
                              professor_id: str,
                              course: str,
                              semester: str,
                              assignment_no: str,
                              question: str,
                              student_answer: str,
                              language: str,
                              old_score: float,
                              new_score: float,
                              old_feedback: str,
                              new_feedback: str) -> int:
        """
        Backward-compatible wrapper that now performs an UPSERT by delegating
        to insert_or_update_grading_result.
        """
        return self.insert_or_update_grading_result(
            student_id=student_id,
            professor_id=professor_id,
            course=course,
            semester=semester,
            assignment_no=assignment_no,
            question=question,
            student_answer=student_answer,
            language=language,
            old_score=old_score,
            new_score=new_score,
            old_feedback=old_feedback,
            new_feedback=new_feedback
        )

    # ------------------------------------------------------------------
    # Versioning / corrections
    # ------------------------------------------------------------------
    def insert_grading_correction(self,
                                  student_id: str,
                                  professor_id: str,
                                  assignment_no: str,
                                  question: str,
                                  old_score: float,
                                  new_score: float,
                                  old_feedback: str,
                                  new_feedback: str,
                                  editor_id: str,
                                  language: str) -> None:
        """
        Log a correction entry for version tracking, including language.
        """
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO grading_corrections (
                    student_id, professor_id, assignment_no, question,
                    old_score, new_score, old_feedback, new_feedback,
                    editor_id, language, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    student_id,
                    professor_id,
                    assignment_no,
                    question,
                    old_score,
                    new_score,
                    old_feedback,
                    new_feedback,
                    editor_id,
                    language,
                    datetime.now()
                )
            )
        self.conn.commit()

    def update_grading_result_with_correction(self,
                                              grading_result_id: int,
                                              new_score: float,
                                              new_feedback: str,
                                              editor_id: str) -> None:
        """
        Save the old result into corrections and update the main grading_results row.
        """
        self.connect()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, student_id, professor_id, assignment_no, question,
                       language, new_score, new_feedback
                  FROM grading_results
                 WHERE id = %s;
                """,
                (grading_result_id,)
            )
            record = cur.fetchone()
            if not record:
                raise ValueError(f"Grading result id {grading_result_id} not found")

            # Log the correction
            self.insert_grading_correction(
                student_id=     record['student_id'],
                professor_id=   record['professor_id'],
                assignment_no=  record['assignment_no'],
                question=       record['question'],
                old_score=      record['new_score'],
                new_score=      new_score,
                old_feedback=   record['new_feedback'],
                new_feedback=   new_feedback,
                editor_id=      editor_id,
                language=       record['language']
            )

            # Apply the update
            cur.execute(
                """
                UPDATE grading_results
                   SET new_score    = %s,
                       new_feedback = %s,
                       created_at   = %s
                 WHERE id = %s;
                """,
                (new_score, new_feedback, datetime.now(), grading_result_id)
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------
    def fetch_results(self, filters: dict = None) -> list:
        """
        Fetch grading results, applying optional filters by professor_id, course, semester,
        assignment_no, student_id, or language. Always returns the 'id' field for sharing.
        """
        self.connect()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = (
                "SELECT id, student_id, professor_id, course, semester,"
                " assignment_no, question, student_answer, language,"
                " old_score, new_score, old_feedback, new_feedback, created_at"
                " FROM grading_results WHERE 1=1"
            )
            params = []
            if filters:
                pid = filters.get('professor_id')
                if pid and pid != 'All':
                    query += " AND professor_id = %s"
                    params.append(pid)
                for fld in ("course","semester","assignment_no","student_id","language"):
                    val = filters.get(fld)
                    if val and val != "All":
                        query += f" AND {fld} = %s"
                        params.append(val)

            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        for r in rows:
            try:
                r["score_numeric"] = float(r.get("new_score") or 0)
            except:
                r["score_numeric"] = 0.0
        return rows

    def fetch_my_results(self, professor_email: str, filters: dict = None) -> list:
        """
        Convenience method to fetch only the grading results owned by this professor.
        """
        if filters is None:
            filters = {}
        filters['professor_id'] = professor_email
        return self.fetch_results(filters)

    # ------------------------------------------------------------------
    # Sharing
    # ------------------------------------------------------------------
    def share_result(self, owner_email: str, target_email: str, result_id: int):
        """
        Share a grading result (by its ID) from owner_email to target_email.
        Duplicate shares are simply ignored.
        """
        self.connect()
        with self.conn.cursor() as cur:
            # check if already shared
            cur.execute(
                """
                SELECT 1 FROM result_shares
                 WHERE owner_professor_email = %s
                   AND shared_with_email     = %s
                   AND grading_result_id     = %s;
                """,
                (owner_email, target_email, result_id)
            )
            if not cur.fetchone():
                cur.execute(
                    """
                    INSERT INTO result_shares (
                      owner_professor_email,
                      shared_with_email,
                      grading_result_id,
                      created_at
                    ) VALUES (%s, %s, %s, %s);
                    """,
                    (owner_email, target_email, result_id, datetime.now())
                )
        self.conn.commit()

    def revoke_share(self, owner_email: str, target_email: str, result_id: int):
        """
        Revoke a previously granted share.
        """
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM result_shares
                 WHERE owner_professor_email = %s
                   AND shared_with_email     = %s
                   AND grading_result_id     = %s;
                """,
                (owner_email, target_email, result_id)
            )
        self.conn.commit()

    def fetch_shared_with_me(self, my_email: str) -> list:
        """
        Return all grading_results rows that have been shared with my_email.
        """
        self.connect()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT gr.*, rs.owner_professor_email AS shared_by
                  FROM grading_results gr
                  JOIN result_shares rs
                    ON gr.id = rs.grading_result_id
                 WHERE rs.shared_with_email = %s;
                """,
                (my_email,)
            )
            rows = cur.fetchall()

        for r in rows:
            try:
                r["score_numeric"] = float(r.get("new_score") or 0)
            except:
                r["score_numeric"] = 0.0
        return rows

    def fetch_my_shares(self, owner_email: str) -> list:
        """
        Return all shares I (owner_email) have created.
        """
        self.connect()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                  rs.grading_result_id AS result_id,
                  rs.shared_with_email  AS shared_with,
                  rs.created_at
                FROM result_shares rs
                WHERE rs.owner_professor_email = %s
                ORDER BY rs.created_at DESC;
                """,
                (owner_email,)
            )
            shares = cur.fetchall()
        return shares
