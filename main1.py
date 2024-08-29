def generate_sql_query(user_input, conversation_history):
    """Generate a SQL query based on the user's input."""
    reference_logic = """
    Logic from reference query:
    1. Use a CTE (Common Table Expression) named 'cte' for complex calculations. Use scaler_ebdb_mit_enums table to get the enum value of any table where table_identifier column has tbale name and column_identifier column has the column name of corresponding table
    2. Join these main tables: mentee_lessons, batch_lessons, lessons, batches, super_batches, super_batch_groups, academy_topics, academy_modules, super_batch_academy_topics, courses, mentees, mentee_modules, mentee_batches, users, interviewbit_tests, interviewbit_test_problems, problems, interviewbit_test_sessions, interviewbit_test_session_problems.
    3. Calculate program using CASE statement based on course slug (case when c.slug in ('scaler-wp-intake','scaler-wp-elitex','scaler-wp-elitex-slow','scaler-wp-superx','scaler-wp-beginner','scaler-wp-beginner-refresher',
        'scaler-wp-beginner-repeater-failed-once','scaler-wp-beginner-repeater-failed-twice','scaler-wp-data-engineering','scaler-wp-elitex-refresher-java','scaler-wp-elitex-refresher-python',
        'scaler-wp-fullstack','scaler-wp-backend')
        then 'Academy'
    when c.slug in ('scaler-ds-intake','scaler-ds-elitex','scaler-ds-elitex-refresher','scaler-ds-superx','scaler-ds-beginner','scaler-ds-beginner-refresher',
        'scaler-ds-analytics','scaler-ds-beginner-without-python')
        then 'DSML'
    when c.slug in ('scaler-elitex', 'scaler-superx', 'scaler-plus', 'techversity-beginner', 'techversity-advanced')
        then 'Old Academy'
    when c.slug in ('scaler-wp-us-intake','scaler-wp-us-elitex','scaler-wp-us-superx','scaler-wp-us-beginner', 'scaler-crash-course',
        'scaler-ug-primary')
        then 'US Academy'
    else c.slug 
    end) as program
    4. Always convert datetime columns to date type
    6. Calculate class_type using CASE statement based on lesson_type and lecture_bucket_id.
    7. Calculate live_attendance, recorded_attendance, and overall_attendance using LEAST and MAX functions.
    8. Count assignment and homework problems (total and solved) based on test_type.
    9. Calculate assignment_psp and hw_psp as percentages of solved problems.
    10. Exclude Scaler and InterviewBit learner emails.
    11. Group by user_id, email, batch name, super_batch_name, program, module_name, order, class_topic, class_date, class_timing, and lecture_bucket_id.
    12. Order by email, module order, and class_date.
    13. Do not include any default filters or exclusions unless specified in the user's request.
    14. Whenever the batch name is needed use the name from the super_batches table.
    15. whenever class date is needed use super batch academy topics table and column name date_of_topic. And remember to join super batch academy topics table use academy topic id and super batch id 
    16. Aggregates are not allowed in WHERE clause
    17. TYPECAST THE DATETIME COLUMNS TO DATE
    18. always remember when any of these are used apply this formula in case of learner level data
    ,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS live_attendance
	,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.passive_attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS recorded_attendance
    ,LEAST(MAX(CASE 
				WHEN l.lesson_type in (0,6)
					THEN COALESCE(ml.aggregate_attendance, 0)
				END) / LEAST(MAX(bl.adjustment_factor), 0.8), 100) AS overall_attendance
	,count(DISTINCT CASE 
			WHEN ibt.test_type = 2
				THEN ibtp.id
			END) AS total_assignment_problems
	,count(DISTINCT CASE 
			WHEN ibtsp.STATUS = 2
				AND ibt.test_type = 2
				THEN ibtsp.id
			END) AS total_assignment_problems_solved
	,CASE 
		WHEN total_assignment_problems != 0
			THEN ((total_assignment_problems_solved * 1.00) / total_assignment_problems) * 100
		END AS assignment_psp
	,count(DISTINCT CASE 
			WHEN ibt.test_type = 3
				THEN ibtp.id
			END) AS total_hw_problems
	,count(DISTINCT CASE 
			WHEN ibtsp.STATUS = 2
				AND ibt.test_type = 3
				THEN ibtsp.id
			END) AS total_hw_problems_solved
	,CASE 
		WHEN total_hw_problems != 0
			THEN ((total_hw_problems_solved * 1.00) / total_hw_problems) * 100
		END AS hw_psp
    19. REMEMBER WHENEVER THERE IS SOME AGGREGATE DATA ASKED WHETHER BE ON CLASS LEVEL OR LEARNER LEVEL OR BATCH LEVEL USE LEARNER LEVEL DATA AS BASE DATA.
    21. class name is always as academy topics table with column name title, module name is always academy modules table with column name name
    22. ALWAYS CONVERT COLUMN WITH DATA TYPE AS DATETIME TO DATE DATA TYPE USING ::DATE
    23. REMEMBER TO KEEP THE SYNTAX AS REDSHIFT QUERY
    24. HERE IS DETAIL OF HOW TO CONNECT THE TABLES 
    TABLE MENTEE LESSONS IS CONNECTED WITH MENTEES USING MENTEE ID 
    TABLE BATCH LESSONS IS CONNECTED WITH MENTEE LESSONS USING BATCH LESSON ID
    TABLE LESSONS IS CONNECTED WITH BATCH LESSONS USING LESSON ID
    TABLE ACADEMY TOPICS IS CONNECTED WITH LESSONS USING ACADEMY TOPIC ID
    TABLE BATCHES IS CONNECTED WITH BATCH LESSONS USING BATCH ID 
    TABLE SUPER BATCHES USED FOR GETTING BATCH NAME IS CONNECTED WITH BATCHES USING BATCH ID 
    TABLE SUPER BATCH ACADEMY TOPICS IS CONNECTED USING SUPER BATCH ID IN SUPER BATCHES TABLE AND ID IN ACADEMY TOPICS TABLE 
    WHENEVER IN WHERE CLAUSE DATE IS USED THEN DO DATE_COLUMN::DATE= CONDITION
    25. ALWAYS MAKE THE QUERY IN THIS ORDER :
        mentees -> mentee lessons -> batch lessons -> batches -> super batches -> lessons -> academy topics -> super batch academy topics
    26. ALWAYS USE STATUS ACCORDING TO SCALER_EBDB_MIT_ENUMS TABLE
    27. FOR ASSIGNMENT PSP OR ANY PROBLEM SOLVING METRIC USE
    LEFT JOIN scaler_ebdb_interviewbit_tests ibt ON ibt.id = l.test_id
    LEFT JOIN scaler_ebdb_interviewbit_test_problems ibtp ON ibtp.test_id = ibt.id
    LEFT JOIN scaler_ebdb_interviewbit_test_sessions ibts ON ibts.test_id = ibt.id
	AND users.id = ibts.user_id
    LEFT JOIN scaler_ebdb_interviewbit_test_session_problems ibtsp ON ibtsp.test_session_id = ibts.id
    28. WHENEVER ASSIGNMENT PSP OR HW PSP IS ASKED CALCULATE THE total_assignment_problems_solved AND total_assignment_problems FIRSTLY AND AFTER THAT GIVE THE PERCENTAGE
    29. In where clause do not use any subquery
    30. To know about how many paid learners come in a month use 
    punched_leads-> applicant_tracks (using applicant_track_id column from punched_leads) -> super_batches (using super_batch_id from applicant_tracks) -> super_batch_groups (using super_batch_group_id from super_batches) -> cohorts (using cohort_id from super_batch_groups) and alweays use cohort name as filter
    The query to get the data is select c.name as cohort_name,count(distinct pl.lead_email) as learner_count from scaler_ebdb_punched_leads pl join scaler_ebdb_applicant_tracks at on at.id=pl.applicant_track_id join scaler_ebdb_super_batches sb on sb.id=at.super_batch_id join scaler_ebdb_super_batch_groups sbg on sbg.id=sb.super_batch_group_id join scaler_ebdb_cohorts c on c.id=sbg.cohort_id group by c.name,c.end_date order by c.end_date desc;
    31. To know about Net Promoter Score (NPS) and to get nps formula is (promter-detractor)/(promoter+neutral+detractor)*100 
    where promoter is 9 or 10, neutral is 7 or 8 and detractor is 1 to 6 use this table scaler_ebdb_interviewbit_form_responses
    use this query to get the data
    select distinct m.id as responder_id,scaler_ebdb_users.email as email,scaler_ebdb_interviewbit_form_responses.response::int as nps_rating,scaler_ebdb_interviewbit_form_responses.created_at::date as nps_date
FROM scaler_ebdb_interviewbit_form_responses
LEFT JOIN scaler_ebdb_mentee_batches ON scaler_ebdb_mentee_batches.mentee_id = scaler_ebdb_interviewbit_form_responses.responder_id
LEFT JOIN scaler_ebdb_batches ON scaler_ebdb_batches.id = scaler_ebdb_mentee_batches.batch_id
LEFT JOIN scaler_ebdb_super_batches ON scaler_ebdb_super_batches.id = scaler_ebdb_batches.super_batch_id
LEFT JOIN scaler_ebdb_super_batch_groups ON scaler_ebdb_super_batch_groups.id = scaler_ebdb_super_batches.super_batch_group_id
LEFT JOIN scaler_ebdb_mentees m on m.id=scaler_ebdb_interviewbit_form_responses.responder_id
LEFT JOIN scaler_ebdb_users on scaler_ebdb_users.id=m.user_id 
WHERE scaler_ebdb_interviewbit_form_responses.form_id = 112704
	AND scaler_ebdb_interviewbit_form_responses.responder_type = 'Mentee'
	AND extract(day FROM scaler_ebdb_interviewbit_form_responses.created_at) BETWEEN 5
		AND 26
	order by 4 desc 
28. ALWAYS USE DISTINCT KEYWORD IN THE QUERY ANS AVOID USING ALIAS
29. For the bda use owner id in punched leads table and connect it with users table using owner id of punched leads and user id in users table
30. to calculate class rating use SUM(bl.ratings_count) / (COUNT(bl.id) / COUNT(DISTINCT bl.id)) AS ratings_count,
     ROUND(SUM(bl.average_rating * bl.ratings_count)/SUM(bl.ratings_count), 2) AS class_rating
31. To get instructor information use 
join scaler_ebdb_sbat_instructor_associations sia on sia.sbat_id=scaler_ebdb_super_batch_academy_topics.id 
join scaler_ebdb_instructors i ON i.id = sia.instructor_id
join scaler_ebdb_users iu ON iu.id = i.user_id 
32. Dont use scaler_ebdb_mit_enums table unnecessary and when required use enum_val column along with table_identifier and column_identifier

    """

    schema_info = """
    Available tables and their relevant columns:
    - scaler_ebdb_mentee_lessons: id, mentee_id, batch_lesson_id, status, attendance, passive_attendance, aggregate_attendance, lesson_rating, zoom_link, submitted_feedback, events, zoom_identifier
    - scaler_ebdb_batch_lessons: id, batch_id, lesson_id, status, adjustment_factor, start_time, end_time, average_rating, ratings_count, zoom_id, test_link, test_id, summary, archived_session_link, zoom_recordings_id, master_ta_id
    - scaler_ebdb_lessons: id, academy_topic_id, lesson_type, test_id, name, description, duration, course_id, status, created_by, updated_by, meta
    - scaler_ebdb_batches: id, super_batch_id, batch_name, channel_id, scheduler
    - scaler_ebdb_super_batches: id, name, course_id, super_batch_group_id, start_date, duration, class_start_time, timezone, status, country, ta_flock_channel_id, announcement_channel_id, meta, onboarding_details, timeline, eligibility_info, min_grad_year, max_grad_year, min_experience, max_experience, academy_schedule_id, batch_type, sequence
    - scaler_ebdb_super_batch_groups: id, group_name, cohort_id, meta
    - scaler_ebdb_academy_topics: id, title, academy_module_id, slug, level, course_id, type_of_topic, parent_topic_id, updated_by, created_by, bucket_id, activity_type, junction_number
    - scaler_ebdb_academy_modules: id, name, category, duration_in_weeks, dummy
    - scaler_ebdb_super_batch_academy_topics: super_batch_id, academy_topic_id, date_of_topic
    - scaler_ebdb_courses: id, title, slug
    - scaler_ebdb_mentees: id, user_id, status, solved_percentage
    - scaler_ebdb_mentee_modules: id, mentee_id, academy_module_id, status, order
    - scaler_ebdb_mentee_batches: mentee_id, batch_id, status, current_academy_streak, streak_updated_at, best_streak, streak_freeze_count, degree_enrollment_status, user_id
    - scaler_ebdb_users: id, email, name, username, slug, phone_number
    - scaler_ebdb_interviewbit_tests: id, test_type, name, duration, status, created_by, updated_by, meta
    - scaler_ebdb_interviewbit_test_problems: id, test_id, problem_id, order, max_score, custom_question_name, lock, message
    - scaler_ebdb_problems: id, problem_statement, judge_type
    - scaler_ebdb_interviewbit_test_sessions: id, test_id, user_id, status, start_time, end_time, score, meta, candidate_name, candidate_email, candidate_work_experience, candidate_gender, candidate_contact_number, candidate_city, candidate_cgpa, candidate_degree, candidate_branch, candidate_university, candidate_graduation_city, disclaimer, user_start_time, user_url_identifier, restricted_events, owner_session, opened_at, clicked_at, slug, platform_feedback, question_feedback, practice_score
    - scaler_ebdb_interviewbit_test_session_problems: id, test_session_id, problem_id, status, score, start_time, time_to_solve, first_solved_at, practice_score, penalty_time
    - scaler_ebdb_companies: id, name, slug, use_dashboard
    - scaler_ebdb_company_team_members: id, company_team_id, user_id, member_type, access_levels
    - scaler_ebdb_company_teams: id, name, company_id, allocated, state
    - scaler_ebdb_interviewbit_test_accounts: id, company_id, email, company_name
    - scaler_ebdb_problem_tests: id, problem_id, score, time_limit, memory_limit, created_at, updated_at, input_data_file_name, output_data_file_name, input_lines_per_test, test_cases_count
    - scaler_ebdb_interviewbit_test_session_codes: id, test_session_id, problem_id, programming_language_id, user_saved_code_id, session_type
    - scaler_ebdb_interviewbit_test_user_submissions: id, test_id, test_session_id, problem_id, user_submission_id, submission_type
    - scaler_ebdb_user_submissions: id, user_id, problem_id, submission_type, programming_language_id, submission_content, status, result
    - scaler_ebdb_user_submission_test_results: id, user_submission_id, user_id, problem_test_id, result, score, time, memory, first_failed_test_case, failed_input, expected_output, user_output, error_message, created_at, updated_at
    - scaler_ebdb_user_saved_codes: id, user_id, problem_id, programming_language_id, content
    - scaler_ebdb_course_modules: id, course_id, academy_module_id
    - scaler_ebdb_super_batch_modules: id, supere_batch_id, academy_module_id
    - scaler_ebdb_mentee_session_grants: id, mentee_id, category, grant_type, owner_type, owner_id
    - scaler_ebdb_ninja_hire_interviews: id, type
    - scaler_ebdb_ninja_hire_associations: ninja_hire_interview_id, associate
    - scaler_ebdb_notebooks: id, name, directory_type, parent_id
    - scaler_ebdb_notebook_owner: id, user_id, notebook_id, access
    - scaler_ebdb_notebook_resources: id, resource_type, resource_id, notebook_id
    - scaler_ebdb_notebook_associations: id, associate_type, associate_id, notebook_id
    - scaler_ebdb_notes: id, content
    - scaler_ebdb_punched_leads: id, applicant_track_id, lead_name, lead_email, lead_mobile, lead_source, company_type, owner_id, cycle, current_college, current_company, current_ctc, total_work_experience, current_designation, down_payment_amount, cycle_start_date, cycle_end_date, sale_amount, sale_status, enrollment_type
    - scaler_ebdb_applicant_tracks: id, user_id, status, test_id, applicant_type, super_batch_id, submitted_on, type, available_from, referred_by, rejection_reason, source, medium, campaign, meta, owner_id, status_updated_at, status_expiring_at
    - scaler_ebdb_cohorts: id, name, start_date, end_date, course, meta_info
    - scaler_ebdb_interviewbit_form_groups: id, name, description, status, created_by, updated_by
    - scaler_ebdb_interviewbit_forms: id, name, description, form_json, status, created_by, updated_by
    - scaler_ebdb_interviewbit_form_responses: id, form_id, user_id, response_json
    """

    prompt = f"""Given the following reference logic, schema information, and conversation history:

Reference Logic:
{reference_logic}

Schema Information:
{schema_info}

Conversation History:
{conversation_history}

Generate a SQL query for the following request: {user_input}

Return only the SQL query, without any explanations, comments, or formatting. Use the DISTINCT keyword where appropriate. Follow the structure and logic of the reference query, adapting it to the specific request. Learn from any feedback or corrections in the conversation history."""

    generated_sql = get_gpt4_response(prompt, conversation_history)
    return clean_sql_query(generated_sql)
