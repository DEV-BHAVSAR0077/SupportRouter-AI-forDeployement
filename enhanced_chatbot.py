import requests
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
import chromadb

class EnhancedDepartmentRouterChatbot:
    def __init__(self, api_key: str, chroma_db_path: str = "./chroma_db"):
        """Initialize the enhanced chatbot with Gemini API and ChromaDB."""
        # Gemini AI setup
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-flash-latest')
        
        # ChromaDB Cloud setup
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
        
        # Create or get collections
        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"description": "All chat conversations"}
        )
        self.knowledge_base_collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"description": "Company knowledge base and FAQs"}
        )
        
        # Session state
        self.chat_history: List[Dict[str, str]] = []
        self.session_id = str(uuid.uuid4())
        self.identified_department: Optional[str] = None
        self.department_confidence: int = 0
        
        # Email workflow state
        self.email_workflow_active = False
        self.email_confirmation_pending = False
        self.original_user_question = ""
        self.bot_response_to_question = ""
        self.email_data = {}
        self.current_step = 0
        self.workflow_steps = ["name", "email", "company", "additional_details"]

        # Department configuration with email addresses
        self.departments = {
            "Sales": {
                "description": "Project inquiries, pricing, proposals, and new business discussions",
                "email": os.getenv("SALES_EMAIL")
            },
            "Technical Support": {
                "description": "Website issues, hosting, performance, bugs, and technical assistance",
                "email": os.getenv("SUPPORT_EMAIL")
            },
            "Customer Service": {
                "description": "Client communication, service concerns, and ongoing project support",
                "email": os.getenv("CUSTOMER_SERVICE_EMAIL")
            },
            "HR": {
                "description": "Careers, recruitment, interviews, and internal policies",
                "email": os.getenv("HR_EMAIL")
            },
            "Billing": {
                "description": "Invoices, payments, contracts, and account-related queries",
                "email": os.getenv("BILLING_EMAIL")
            },
            "General Inquiry": {
                "description": "Partnerships, digital services, and general information",
                "email": os.getenv("GENERAL_INQUIRY_EMAIL")
            }
        }
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = os.getenv("SMTP_PORT")
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
    
    def analyze_with_gemini(self, user_message: str, conversation_context: str, knowledge_context: str = "") -> Dict:
        """Use Gemini API to analyze the user's message and detect department."""
        
        prompt = f"""You are an intelligent support routing assistant for Webential - a digital solutions company.

Analyze the user's message and determine:
1. Which department should handle this request based on the department descriptions and the user's Semantic intent.
2. The importance/urgency level (Low, Medium, High, Critical)
3. A helpful response to the user's question

Available Departments:
{json.dumps(self.departments, indent=2)}

Conversation History:
{conversation_context}

Relevant Knowledge:
{knowledge_context}

User Message: {user_message}

Respond in JSON format ONLY:
{{
    "department": "Department Name",
    "confidence": 85,
    "importance": "High/Medium/Low/Critical",
    "user_response": "Helpful response to the user's question",
    "reason": "Brief explanation for department choice"
}}"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                return {
                    "department": "General Inquiry",
                    "confidence": 50,
                    "importance": "Medium",
                    "user_response": "I'll help you with your inquiry. Let me route this to the appropriate team.",
                    "reason": "Unable to parse AI response"
                }
        except Exception as e:
            print(f"Error analyzing with Gemini: {e}")
            return {
                "department": "General Inquiry",
                "confidence": 50,
                "importance": "Medium",
                "user_response": "Thank you for your message. I'll make sure the right team looks into this.",
                "reason": f"Error: {str(e)}"
            }
    
    def process_workflow_step(self, user_input: str) -> Dict:
        """Process the simplified email workflow steps."""
        current_step_name = self.workflow_steps[self.current_step]
        
        if current_step_name == "name":
            self.email_data["name"] = user_input.strip()
            self.current_step += 1
            return {
                "response": "Thank you! Now, please provide your **email address**:",
                "complete": False
            }
        
        elif current_step_name == "email":
            # Validate email format
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            email_match = re.search(email_pattern, user_input.strip())
            
            if email_match:
                self.email_data["email"] = email_match.group(0)
                self.current_step += 1
                return {
                    "response": "Great! Please provide your **company name** (or type 'individual' if personal):",
                    "complete": False
                }
            else:
                return {
                    "response": "Please provide a valid email address (e.g., name@company.com):",
                    "complete": False
                }
        
        elif current_step_name == "company":
            self.email_data["company"] = user_input.strip()
            self.current_step += 1
            return {
                "response": f"""Perfect! Do you have any **additional details** you'd like to add regarding your question?

**Your original question:** "{self.original_user_question}"

(Type any additional information, or type 'none' if you don't have any)""",
                "complete": False
            }
        
        elif current_step_name == "additional_details":
            self.email_data["additional_details"] = user_input.strip() if user_input.lower() not in ['none', 'n/a', 'no'] else ""
            
            # Generate and send email
            email_html = self.generate_intelligent_email()
            email_sent = self.send_email_to_department(email_html)
            
            self.email_workflow_active = False
            
            response_msg = "✅ **Email Generated and Sent Successfully!**\n\n"
            if email_sent:
                response_msg += f"Your request has been forwarded to our **{self.identified_department}** team. They will contact you at **{self.email_data.get('email')}** shortly."
            else:
                response_msg += f"⚠️ Email generated but could not be sent automatically (SMTP not configured).\n\nThe email has been prepared and will be sent to the **{self.identified_department}** department."
            
            return {
                "response": response_msg,
                "email_html": email_html,
                "complete": True,
                "email_sent": email_sent
            }
        
        return {
            "response": "Something went wrong. Please try again.",
            "complete": False
        }
    
    def generate_intelligent_email(self) -> str:
        """Generate professional email based on collected data and AI analysis."""
        
        # Determine importance level
        importance = self.email_data.get("importance", "Medium")
        
        # Generate email content using AI
        prompt = f"""Generate a professional email FROM a customer TO the {self.identified_department} department.

IMPORTANT: Write the email as if the USER (customer) is writing directly to the department team. Use first person ("I", "my", "we", "our").

User Information (the sender):
- Name: {self.email_data.get('name')}
- Email: {self.email_data.get('email')}
- Company: {self.email_data.get('company')}

User's Question/Issue: {self.original_user_question}

Additional Details from User: {self.email_data.get('additional_details', 'None provided')}

Importance Level: {importance}

Instructions:
1. Write in FIRST PERSON from the user's perspective ("I need help with...", "We are experiencing...", "My website is...")
2. Address the department team directly ("Dear Technical Support Team", "Dear Sales Team")
3. Use a tone appropriate for {importance} importance level:
   - Critical/High: Urgent, direct, emphasize time-sensitivity
   - Medium: Professional, clear, polite
   - Low: Friendly, conversational, relaxed
4. The user is reaching out FOR HELP or information FROM the department
5. Write as if the customer is explaining their situation/question to the support team
6. Include the user's question naturally in the email body
7. Add additional details if provided
8. The writing style should REFLECT the importance without explicitly stating it
9. Use appropriate business email tone (professional but natural)

Example structure:
- "I am writing to request..."
- "We are experiencing..."
- "I would like to inquire about..."
- "My company needs assistance with..."

Respond in JSON format:
{{
    "subject": "Subject line (concise, describes the request)",
    "greeting": "Dear [Department] Team,",
    "paragraph1": "Opening paragraph introducing the issue/question from user's perspective",
    "paragraph2": "Main content with details (use 'I', 'my', 'we', 'our')",
    "paragraph3": "Additional details/closing paragraph if needed",
    "closing": "Appropriate closing (e.g., Sincerely, Thank you, Best regards) based on importance",
    "signature_contact": "Email or phone if applicable"
}}
IMPORTANT: Ensure the email content uses double asterisks for **bolding** key terms (like **Project Name**), which will be formatted correctly."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                email_parts = json.loads(json_match.group())
            else:
                # Fallback - write from user perspective
                email_parts = {
                    "subject": f"Support Request: {self.original_user_question[:50]}",
                    "greeting": f"Dear {self.identified_department} Team,",
                    "paragraph1": f"I am reaching out regarding the following matter:",
                    "paragraph2": self.original_user_question,
                    "paragraph3": f"Additional details: {self.email_data.get('additional_details', 'N/A')}" if self.email_data.get('additional_details') else "",
                    "closing": "Thank you for your assistance,"
                }
        except Exception as e:
            print(f"Error generating email content: {e}")
            email_parts = {
                "subject": f"Support Request: {self.original_user_question[:50]}",
                "greeting": f"Dear {self.identified_department} Team,",
                "paragraph1": "I am reaching out regarding the following matter:",
                "paragraph2": self.original_user_question,
                "paragraph3": f"Additional details: {self.email_data.get('additional_details', 'N/A')}" if self.email_data.get('additional_details') else "",
                "closing": "Thank you for your assistance,"
            }
        
        # Build email body
        body_paragraphs = []
        body_paragraphs.append(email_parts.get('greeting', 'Dear Team,'))
        body_paragraphs.append("")
        
        for i in range(1, 4):
            para = email_parts.get(f'paragraph{i}', '').strip()
            if para:
                body_paragraphs.append(para)
                body_paragraphs.append("")
        
        # Add signature from AI response or fallback
        body_paragraphs.append(email_parts.get('closing', 'Best regards,'))
        
        # If AI provided a signature, use it. Otherwise append name manually.
        if 'signature_name' in email_parts:
            body_paragraphs.append(email_parts['signature_name'])
            if 'signature_contact' in email_parts:
                body_paragraphs.append(email_parts['signature_contact'])
        else:
            # Fallback if AI didn't provide signature structure
            body_paragraphs.append(self.email_data.get('name', 'User'))
            if self.email_data.get('company') and self.email_data['company'].lower() != 'individual':
                body_paragraphs.append(self.email_data['company'])
        
        email_body_text = '\n'.join(body_paragraphs)
        
        # Convert Markdown bold to HTML bold
        email_body_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', email_body_text)
        
        # Create professional HTML email for department
        department_email_html = self._create_department_email_html(
            email_parts.get('subject', 'New Inquiry'),
            email_body_text
        )
        
        # Also create user-facing HTML preview
        user_preview_html = self._create_user_preview_html(
            email_parts.get('subject', 'New Inquiry'),
            email_body_text
        )
        
        # Store both
        self.email_data['department_email'] = department_email_html
        self.email_data['user_preview'] = user_preview_html
        self.email_data['subject'] = email_parts.get('subject', 'New Inquiry')
        
        return user_preview_html
    
    def _create_department_email_html(self, subject: str, body: str) -> str:
        """Create the professional email template for departments."""
        
        user_details = {
            'name': self.email_data.get('name', 'Anonymous User'),
            'email': self.email_data.get('email', 'not_provided@email.com'),
            'company': self.email_data.get('company', 'Not provided')
        }
        
        html = f"""<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #000000;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #1e3a8a;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .header h2 {{
                margin: 0;
                font-family: 'Times New Roman', Times, serif;
                font-size: 18pt;
            }}
            .metadata {{
                background-color: #f3f4f6;
                padding: 15px;
                border-left: 4px solid #1e3a8a;
                margin-bottom: 20px;
                font-family: 'Times New Roman', Times, serif;
            }}
            .metadata p {{
                margin: 5px 0;
                font-size: 11pt;
            }}
            .content {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 12pt;
                white-space: pre-wrap;
                line-height: 1.8;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 15px;
                border-top: 1px solid #d1d5db;
                font-size: 10pt;
                color: #6b7280;
                font-family: 'Times New Roman', Times, serif;
            }}
            hr {{
                border: none;
                border-top: 2px solid #e5e7eb;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>New Support Request - {self.identified_department} Department</h2>
        </div>
        
        <div class="metadata">
            <p><strong>From:</strong> {user_details['name']}</p>
            <p><strong>Email:</strong> {user_details['email']}</p>
            <p><strong>Company/Organization:</strong> {user_details['company']}</p>
            <p><strong>Session ID:</strong> {self.session_id}</p>
            <p><strong>Date & Time:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <hr>
        
        <div class="content">
{body}
        </div>
        
        <div class="footer">
            <p>This message was sent via Webential Support Desk System</p>
            <p>Please reply directly to the user's email address provided above.</p>
        </div>
    </body>
</html>"""
        
        return html
    
    def _create_user_preview_html(self, subject: str, body: str) -> str:
        """Create user-friendly preview of the email."""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000000;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #ffffff;
        }}
        .email-subject {{
            font-weight: bold;
            margin-bottom: 25px;
            font-size: 12pt;
        }}
        .email-content {{
            margin-top: 20px;
            white-space: pre-line;
        }}
        hr {{
            border: none;
            border-top: 1px solid #cccccc;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="email-subject">
        <strong>Subject:</strong> {subject}
    </div>
    <hr>
    <div class="email-content">
{body}
    </div>
</body>
</html>"""
        
        return html
    
    # def send_email_to_department(self, email_html: str) -> bool:
    #     """Send the email to the identified department."""
        
    #     if not self.sender_email or not self.sender_password:
    #         print("⚠️  Email credentials not configured.")
    #         print(f"Would send to: {self.departments[self.identified_department]['email']}")
    #         return False
        
    #     try:
    #         recipient_email = self.departments[self.identified_department]["email"]
    #         subject = self.email_data.get('subject', f'New Request - {self.identified_department}')
            
    #         # Create message
    #         message = MIMEMultipart("alternative")
    #         message["Subject"] = subject
    #         message["From"] = self.sender_email
    #         message["To"] = recipient_email
    #         message["Reply-To"] = self.email_data.get('email', self.sender_email)
            
    #         # Use the department email HTML
    #         department_html = self.email_data.get('department_email', email_html)
    #         part = MIMEText(department_html, "html")
    #         message.attach(part)
            
    #         # Send email
    #         with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
    #             server.starttls()
    #             server.login(self.sender_email, self.sender_password)
    #             server.sendmail(self.sender_email, recipient_email, message.as_string())
            
    #         print(f"✅ Email sent successfully to {recipient_email}")
    #         return True
            
    #     except Exception as e:
    #         print(f"❌ Error sending email: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False

    def send_email_to_department(self, email_html: str) -> bool:
        """Send the email to the identified department using Resend API."""

        try:
            api_key = os.getenv("RESEND_API_KEY")
            if not api_key:
                raise Exception("RESEND_API_KEY not configured")

            recipient_email = self.departments[self.identified_department]["email"]
            subject = self.email_data.get(
                "subject",
                f"New Request - {self.identified_department}"
            )

            # Use department-specific HTML if available
            department_html = self.email_data.get(
                "department_email",
                email_html
            )

            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": f"Support Bot <{self.sender_email}>",
                    "to": [recipient_email],
                    "subject": subject,
                    "html": department_html,
                    "reply_to": self.email_data.get("email"),
                },
                timeout=10
            )

            response.raise_for_status()

            print(f"✅ Email sent successfully to {recipient_email}")
            return True
        
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
    
    def save_to_chroma(self, message: str, role: str, metadata: Dict = None):
        """Save conversation message to ChromaDB."""
        doc_id = f"{self.session_id}_{len(self.chat_history)}_{role}"
        
        metadata = metadata or {}
        metadata.update({
            "session_id": self.session_id,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "department": self.identified_department or "None"
        })
        
        self.conversations_collection.add(
            documents=[message],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search knowledge base for relevant information."""
        try:
            results = self.knowledge_base_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results and results['documents'] and results['documents'][0]:
                return [
                    {
                        "content": doc,
                        "metadata": meta
                    }
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0])
                ]
            return []
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    def process_message(self, user_message: str) -> Dict:
        """Process user message with streamlined workflow."""
        
        # If email workflow is active, process the current step
        if self.email_workflow_active:
            result = self.process_workflow_step(user_message)
            
            self.chat_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            self.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "bot_response": result["response"],
                "status": "email_complete" if result["complete"] else "collecting_info",
                "workflow_active": not result["complete"],
                "current_step": self.current_step,
                "session_id": self.session_id,
                "email_html": result.get("email_html"),
                "email_sent": result.get("email_sent", False),
                "analysis": {
                    "department": self.identified_department,
                    "confidence": self.department_confidence,
                    "reason": "Email workflow in progress"
                },
                "department": self.identified_department
            }
        
        # If waiting for confirmation to send email
        if self.email_confirmation_pending:
            if any(word in user_message.lower() for word in ['yes', 'yeah', 'sure', 'ok', 'okay', 'please', 'send']):
                # User wants to send email - start workflow
                self.email_workflow_active = True
                self.email_confirmation_pending = False
                self.current_step = 0
                
                response = "Great! Let's collect some information to send your request to our team.\n\n**Please provide your full name:**"
                
                self.chat_history.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                })
                self.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "bot_response": response,
                    "status": "workflow_started",
                    "workflow_active": True,
                    "current_step": 0,
                    "session_id": self.session_id,
                    "analysis": {
                        "department": self.identified_department,
                        "confidence": self.department_confidence,
                        "reason": "User confirmed email sending"
                    },
                    "department": self.identified_department
                }
            else:
                # User doesn't want to send email
                self.email_confirmation_pending = False
                response = "No problem! Is there anything else I can help you with?"
                
                self.chat_history.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                })
                self.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "bot_response": response,
                    "status": "normal",
                    "workflow_active": False,
                    "session_id": self.session_id,
                    "analysis": {
                        "department": self.identified_department,
                        "confidence": self.department_confidence,
                        "reason": "User declined email sending"
                    },
                    "department": self.identified_department
                }
        
        # Normal message processing - Answer question first, then offer email
        
        # Search for relevant knowledge
        knowledge_results = self.search_knowledge_base(user_message)
        knowledge_context = "\n".join([
            f"- {result['content']}" 
            for result in knowledge_results
        ]) if knowledge_results else "No relevant knowledge base entries found."
        
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to ChromaDB
        self.save_to_chroma(user_message, "user")
        
        # Create conversation context
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.chat_history[-5:]
        ])
        
        # Analyze with Gemini - This detects department and generates response
        analysis = self.analyze_with_gemini(user_message, conversation_context, knowledge_context)
        
        # Store the department and question for potential email
        self.identified_department = analysis["department"]
        self.department_confidence = analysis["confidence"]
        self.original_user_question = user_message
        self.email_data["importance"] = analysis.get("importance", "Medium")
        
        # Bot responds to the question
        bot_response = analysis.get("user_response", "I'll help you with that.")
        
        # Add follow-up question about sending email
        email_offer = f"\n\n📧 Would you like me to forward this question to our **{self.identified_department}** team via email for a detailed response?"
        
        full_response = bot_response + email_offer
        
        self.bot_response_to_question = bot_response
        self.email_confirmation_pending = True
        
        # Add bot response to history
        self.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to ChromaDB
        self.save_to_chroma(full_response, "assistant", {
            "department": self.identified_department,
            "confidence": analysis["confidence"]
        })
        
        return {
            "bot_response": full_response,
            "status": "awaiting_confirmation",
            "workflow_active": False,
            "session_id": self.session_id,
            "analysis": {
                "department": analysis["department"],
                "confidence": analysis["confidence"],
                "reason": analysis.get("reason", "")
            },
            "department": analysis["department"]
        }
    
    def reset_conversation(self):
        """Reset conversation and workflow state."""
        self.chat_history = []
        self.session_id = str(uuid.uuid4())
        self.identified_department = None
        self.department_confidence = 0
        self.email_workflow_active = False
        self.email_confirmation_pending = False
        self.original_user_question = ""
        self.bot_response_to_question = ""
        self.email_data = {}
        self.current_step = 0
    
    def add_to_knowledge_base(self, content: str, metadata: Dict):
        """Add entry to database."""
        doc_id = f"kb_{uuid.uuid4()}"
        self.knowledge_base_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about conversations and knowledge base."""
        try:
            conv_count = self.conversations_collection.count()
            kb_count = self.knowledge_base_collection.count()
            
            return {
                "total_conversations": conv_count,
                "knowledge_base_entries": kb_count,
                "current_session": self.session_id,
                "current_department": self.identified_department
            }
        except:
            return {
                "total_conversations": 0,
                "knowledge_base_entries": 0,
                "current_session": self.session_id,
                "current_department": self.identified_department
            }


def check_knowledge_base_status(chatbot: EnhancedDepartmentRouterChatbot) -> Dict:
    try:
        stats = chatbot.get_conversation_stats()
        kb_count = stats.get("knowledge_base_entries", 0)
        conv_count = stats.get("total_conversations", 0)
        kb_count = stats.get("knowledge_base_entries", 0)
        print(f"📊 STATUS: {conv_count} conversations | {kb_count} knowledge entries")
        
        if kb_count > 0:
            print("")
        else:
            print(f"   Knowledge base is empty (0 entries)")
            print(f"   You can add knowledge base entries via the API:")
            print(f"   POST /api/knowledge with content and metadata")
        
        return {
            "status": "active" if kb_count > 0 else "empty",
            "entries": kb_count,
            "collection_name": "knowledge_base"
        }
    except Exception as e:
        print(f"❌ Error checking knowledge base: {e}")
        return {
            "status": "error",
            "entries": 0,
            "error": str(e)
        }


def load_knowledge_from_file(chatbot: EnhancedDepartmentRouterChatbot, filepath: str = "knowledge_base_data.json"):
    try:
        if not os.path.exists(filepath):
            print(f"    Knowledge base file not found: {filepath}")
            print(f"    Knowledge base will use existing ChromaDB data only")
            return {"status": "file_not_found", "loaded": 0}
        
        with open(filepath, 'r') as f:
            knowledge_data = json.load(f)
        
        loaded_count = 0
        for entry in knowledge_data:
            if "content" in entry and "metadata" in entry:
                chatbot.add_to_knowledge_base(entry["content"], entry["metadata"])
                loaded_count += 1
        
        print(f"✅ Loaded {loaded_count} entries from {filepath}")
        return {"status": "success", "loaded": loaded_count}
        
    except Exception as e:
        print(f"❌ Error loading knowledge base from file: {e}")
        return {"status": "error", "loaded": 0, "error": str(e)}