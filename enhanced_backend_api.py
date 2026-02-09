from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
from enhanced_chatbot import EnhancedDepartmentRouterChatbot, check_knowledge_base_status
import traceback
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Store chatbot instances per session
chatbot_sessions = {}

def get_chatbot(session_id):
    """Get or create chatbot instance for session."""
    if session_id not in chatbot_sessions:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        chatbot.session_id = session_id
        chatbot_sessions[session_id] = chatbot
        
        # Check knowledge base status (doesn't add sample data)
        # The chatbot will use whatever is already in ChromaDB
        check_knowledge_base_status(chatbot)
    
    return chatbot_sessions[session_id]

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced Department Router Chatbot API is running",
        "features": [
            "ChromaDB vector storage",
            "Knowledge base search from ChromaDB",
            "Email routing",
            "Context-aware responses",
            "Streamlined workflow"
        ]
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Message is required"
            }), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Get chatbot for this session
        chatbot = get_chatbot(session_id)
        
        # Process the message
        result = chatbot.process_message(message)
        
        # Return enhanced response
        return jsonify({
            "success": True,
            "response": result['bot_response'],
            "analysis": {
                "department": result['analysis']['department'],
                "confidence": result['analysis']['confidence'],
                "reason": result['analysis']['reason']
            },
            "status": result['status'],
            "identified_department": result['department'],
            "session_id": result['session_id'],
            "conversation_history": chatbot.chat_history,
            "workflow_active": result.get('workflow_active', False),
            "current_step": result.get('current_step', 0),
            "email_html": result.get('email_html'),
            "email_sent": result.get('email_sent', False)
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "error": "session_id is required"
            }), 400
        
        if session_id in chatbot_sessions:
            chatbot_sessions[session_id].reset_conversation()
            new_session_id = chatbot_sessions[session_id].session_id
        else:
            new_session_id = str(uuid.uuid4())
        
        return jsonify({
            "success": True,
            "message": "Conversation reset successfully",
            "new_session_id": new_session_id
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/departments', methods=['GET'])
def get_departments():
    """Get list of available departments with email addresses."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "API key not configured"
            }), 500
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        
        departments = []
        for dept_name, dept_info in chatbot.departments.items():
            departments.append({
                "name": dept_name,
                "description": dept_info['description'],
                "email": dept_info['email'],
                "keywords": dept_info.get('keywords', [])
            })
        
        return jsonify({
            "success": True,
            "departments": departments
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/history', methods=['POST'])
def get_history():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "error": "session_id is required"
            }), 400
        
        if session_id in chatbot_sessions:
            chatbot = chatbot_sessions[session_id]
            return jsonify({
                "success": True,
                "history": chatbot.chat_history,
                "identified_department": chatbot.identified_department,
                "session_id": chatbot.session_id
            })
        else:
            return jsonify({
                "success": True,
                "history": [],
                "identified_department": None,
                "session_id": None
            })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall system statistics including ChromaDB knowledge base."""
    try:
        # Create temporary chatbot to get stats
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "API key not configured"
            }), 500
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        stats = chatbot.get_conversation_stats()
        kb_status = check_knowledge_base_status(chatbot)
        
        return jsonify({
            "success": True,
            "stats": stats,
            "active_sessions": len(chatbot_sessions),
            "knowledge_base": kb_status
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/knowledge', methods=['POST'])
def add_knowledge():
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({
                "error": "Content is required"
            }), 400
        
        content = data.get('content')
        category = data.get('category', 'general')
        department = data.get('department', 'General Inquiry')
        source = data.get('source', 'manual_entry')
        tags = data.get('tags', [])
        
        # Create temporary chatbot to add knowledge
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "API key not configured"
            }), 500
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        
        metadata = {
            "category": category,
            "department": department,
            "source": source,
            "added_date": str(uuid.uuid4())
        }
        
        if tags:
            metadata["tags"] = ",".join(tags)
        
        chatbot.add_to_knowledge_base(content, metadata)
        
        # Get updated stats
        new_stats = chatbot.get_conversation_stats()
        
        return jsonify({
            "success": True,
            "message": "Knowledge added successfully to ChromaDB",
            "total_entries": new_stats.get("knowledge_base_entries", 0)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/knowledge/search', methods=['POST'])
def search_knowledge():
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Query is required"
            }), 400
        
        query = data.get('query')
        n_results = data.get('n_results', 5)
        
        # Create temporary chatbot to search
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "API key not configured"
            }), 500
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        results = chatbot.search_knowledge_base(query, n_results)
        
        return jsonify({
            "success": True,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/knowledge/list', methods=['GET'])
def list_knowledge():
    """Get all knowledge base entries from ChromaDB."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "API key not configured"
            }), 500
        
        chatbot = EnhancedDepartmentRouterChatbot(api_key)
        
        # Get all entries (search with empty query returns all)
        all_entries = chatbot.knowledge_base_collection.get()
        
        entries = []
        if all_entries and all_entries.get('documents'):
            for i, doc in enumerate(all_entries['documents']):
                entry = {
                    "id": all_entries['ids'][i] if 'ids' in all_entries else f"entry_{i}",
                    "content": doc,
                    "metadata": all_entries['metadatas'][i] if 'metadatas' in all_entries else {}
                }
                entries.append(entry)
        
        return jsonify({
            "success": True,
            "entries": entries,
            "total": len(entries)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file!")
        print("Please create a .env file with your API key before running the server.")
        exit(1)
    
    print("\n🌐 Server starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)