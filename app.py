import os
import requests
import base64
import boto3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from botocore.exceptions import BotoCoreError, ClientError
import re
import pinecone
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configuración AWS Polly
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Configuración Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "mecanica-vehiculos"

# Inicializar clientes
pc = None
index = None
model = None

def init_pinecone():
    """Inicializar Pinecone y el índice"""
    global pc, index, model
    try:
        if PINECONE_API_KEY:
            # Inicializar Pinecone (versión 2.x)
            pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")
            
            # Verificar si el índice existe
            if INDEX_NAME not in pinecone.list_indexes():
                app.logger.info(f"Creando índice {INDEX_NAME}...")
                pinecone.create_index(
                    name=INDEX_NAME,
                    dimension=384,  # all-MiniLM-L6-v2 tiene 384 dimensiones
                    metric="cosine"
                )
                app.logger.info("Índice creado exitosamente")
            
            index = pinecone.Index(INDEX_NAME)
            
            # Cargar el modelo de embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')
            app.logger.info("Pinecone y modelo de embeddings inicializados correctamente")
            
            # Verificar estadísticas del índice
            stats = index.describe_index_stats()
            app.logger.info(f"Estadísticas del índice: {stats}")
            
            return True
        else:
            app.logger.warning("Credenciales de Pinecone no configuradas")
            return False
    except Exception as e:
        app.logger.error(f"Error inicializando Pinecone: {e}")
        return False

def get_embedding(text):
    """Obtener embedding del texto usando all-MiniLM-L6-v2"""
    try:
        if model is None:
            app.logger.error("Modelo no inicializado")
            return None
        
        # Limpiar y preparar el texto
        clean_text = clean_text_for_embedding(text)
        
        # Generar embedding
        embedding = model.encode(clean_text)
        return embedding.tolist()
    except Exception as e:
        app.logger.error(f"Error obteniendo embedding: {e}")
        return None

def clean_text_for_embedding(text):
    """Limpiar texto para mejor embedding"""
    # Remover caracteres especiales pero mantener términos técnicos
    text = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ°-]', ' ', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    # Convertir a minúsculas pero preservar acrónimos comunes
    preserved_terms = ['RPM', 'PSI', 'GPS', 'ABS', 'ESP', 'ECU', 'OBD', 'DTC']
    for term in preserved_terms:
        text = text.replace(term.lower(), term)
    
    return text

def query_pinecone(query, top_k=5):
    """Consultar Pinecone con una pregunta"""
    try:
        if not index:
            app.logger.error("Índice de Pinecone no disponible")
            return None
            
        # Obtener embedding de la consulta
        query_embedding = get_embedding(query)
        if not query_embedding:
            app.logger.error("No se pudo generar embedding para la consulta")
            return None
        
        app.logger.info(f"Consultando Pinecone con: {query}")
        
        # Consultar Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        app.logger.info(f"Pinecone devolvió {len(results.matches)} resultados")
        for i, match in enumerate(results.matches):
            app.logger.info(f"Match {i}: Score={match.score}, ID={match.id}")
        
        return results
    except Exception as e:
        app.logger.error(f"Error consultando Pinecone: {e}")
        return None

def generate_response(query, context_matches):
    """Generar respuesta usando el contexto de Pinecone con enfoque mecánico"""
    try:
        if not context_matches or not context_matches.matches:
            return """No encontré información específica sobre eso en mi base de conocimientos mecánicos. 

Como mecánico especializado, te recomiendo:
1. Proporciona más detalles sobre el problema (sonidos, síntomas, modelo del vehículo)
2. Indica cuándo comenzó el problema
3. Menciona si hay códigos de error en el tablero

¿Podrías darme más información para ayudarte mejor?"""
        
        # Filtrar matches con score mínimo
        relevant_matches = [match for match in context_matches.matches if match.score > 0.7]
        
        if not relevant_matches:
            return generate_low_confidence_response(query)
        
        # Construir contexto a partir de los matches más relevantes
        context_parts = []
        for match in relevant_matches[:3]:  # Solo los 3 más relevantes
            if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                context_parts.append(match.metadata['text'])
        
        context = "\n".join(context_parts)
        
        # Sistema de respuesta basado en reglas para mecánica
        response = generate_mechanical_response(query, context, relevant_matches[0].score)
        return response
        
    except Exception as e:
        app.logger.error(f"Error generando respuesta: {e}")
        return """Lo siento, hubo un error técnico procesando tu consulta mecánica. 

Por favor, intenta reformular tu pregunta o contacta directamente con un técnico especializado."""

def generate_low_confidence_response(query):
    """Respuesta cuando la confianza es baja"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['motor', 'arranque', 'temperatura']):
        return """Basándome en mi experiencia como mecánico, para problemas de motor necesito más información específica:

**Síntomas comunes a revisar:**
- ¿El motor enciende pero no mantiene el ralentí?
- ¿Hay humo de algún color específico?
- ¿La temperatura sube anormalmente?
- ¿Escuchas ruidos extraños?

**Primeras verificaciones:**
- Nivel de aceite y refrigerante
- Estado de la batería (12.6V en reposo)
- Conexiones eléctricas limpias
- Filtros de aire y combustible

¿Podrías describir exactamente qué síntomas presenta el vehículo?"""
    
    return """Como mecánico especializado, necesito más detalles para darte un diagnóstico preciso.

**Por favor comparte:**
- Marca, modelo y año del vehículo
- Síntomas específicos que observas
- Cuándo ocurre el problema
- Sonidos, olores o señales visuales
- Códigos de error (si los hay)

Mientras tanto, verifica:
- Niveles de fluidos (aceite, refrigerante, frenos)
- Estado de la batería
- Presión de neumáticos

¿Qué problema específico tiene tu vehículo?"""

def generate_mechanical_response(query, context, confidence_score):
    """Generar respuesta específica para mecánica de vehículos con contexto de Pinecone"""
    query_lower = query.lower()
    
    # Extraer información clave del contexto
    key_info = extract_technical_info(context)
    
    # Determinar el tipo de consulta y generar respuesta específica
    if any(word in query_lower for word in ['motor', 'encender', 'arrancar', 'temperatura', 'sobrecalentamiento', 'rpm']):
        return generate_motor_response_with_context(key_info, confidence_score)
    elif any(word in query_lower for word in ['frenos', 'frenar', 'pastillas', 'disco', 'pedal']):
        return generate_brakes_response_with_context(key_info, confidence_score)
    elif any(word in query_lower for word in ['aceite', 'lubricante', 'cambio', 'viscosidad', 'filtro']):
        return generate_oil_response_with_context(key_info, confidence_score)
    elif any(word in query_lower for word in ['batería', 'eléctrico', 'corriente', 'alternador', 'voltaje']):
        return generate_electrical_response_with_context(key_info, confidence_score)
    elif any(word in query_lower for word in ['llantas', 'neumáticos', 'presión', 'inflado', 'desgaste']):
        return generate_tires_response_with_context(key_info, confidence_score)
    elif any(word in query_lower for word in ['transmisión', 'caja', 'cambios', 'embrague', 'automática']):
        return generate_transmission_response_with_context(key_info, confidence_score)
    else:
        return generate_general_response_with_context(key_info, query, confidence_score)

def extract_technical_info(context):
    """Extraer información técnica específica del contexto"""
    technical_data = {
        'procedures': [],
        'specifications': [],
        'warnings': [],
        'tools': []
    }
    
    lines = context.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identificar procedimientos
        if any(keyword in line.lower() for keyword in ['paso', 'procedimiento', 'instrucción', 'método']):
            technical_data['procedures'].append(line)
        
        # Identificar especificaciones técnicas
        elif any(keyword in line.lower() for keyword in ['especificación', 'tolerancia', 'medida', 'valor', 'psi', 'rpm', 'voltios']):
            technical_data['specifications'].append(line)
        
        # Identificar advertencias
        elif any(keyword in line.lower() for keyword in ['advertencia', 'precaución', 'peligro', 'importante', 'nota']):
            technical_data['warnings'].append(line)
        
        # Identificar herramientas
        elif any(keyword in line.lower() for keyword in ['herramienta', 'equipo', 'llave', 'medidor']):
            technical_data['tools'].append(line)
    
    return technical_data

def generate_motor_response_with_context(tech_info, confidence):
    """Respuesta específica para motor con contexto"""
    response = "**DIAGNÓSTICO DE MOTOR - Miguel Mecánico**\n\n"
    
    if tech_info['specifications']:
        response += "**Especificaciones técnicas:**\n"
        for spec in tech_info['specifications'][:3]:
            response += f"• {spec}\n"
        response += "\n"
    
    if tech_info['procedures']:
        response += "**Procedimiento recomendado:**\n"
        for i, proc in enumerate(tech_info['procedures'][:3], 1):
            response += f"{i}. {proc}\n"
        response += "\n"
    
    response += """**Verificaciones básicas del motor:**
• Nivel de aceite (entre MIN y MAX en la varilla)
• Temperatura del refrigerante (80-90°C operativo)
• Presión de aceite (2-4 bar a 2000 RPM)
• Estado de filtros (aire, combustible, aceite)

**Síntomas de alerta:**
• Luz de temperatura en tablero
• Ruidos metálicos o golpeteo
• Pérdida de potencia
• Consumo excesivo de combustible"""
    
    if tech_info['warnings']:
        response += "\n\n**⚠️ PRECAUCIONES IMPORTANTES:**\n"
        for warning in tech_info['warnings'][:2]:
            response += f"• {warning}\n"
    
    return response

def generate_brakes_response_with_context(tech_info, confidence):
    """Respuesta específica para frenos con contexto"""
    response = "**SISTEMA DE FRENOS - Miguel Mecánico**\n\n"
    
    if tech_info['procedures']:
        response += "**Procedimiento técnico:**\n"
        for i, proc in enumerate(tech_info['procedures'][:3], 1):
            response += f"{i}. {proc}\n"
        response += "\n"
    
    response += """**Inspección de frenos:**
• Espesor de pastillas (mínimo 3mm)
• Estado de discos (sin ranuras profundas)
• Nivel de líquido de frenos (DOT 3 o DOT 4)
• Flexibilidad de mangueras

**Señales de mantenimiento:**
• Chirrido al frenar (pastillas gastadas)
• Vibración en pedal (discos deformados)
• Pedal esponjoso (aire en sistema)
• Distancia de frenado aumentada"""
    
    if tech_info['specifications']:
        response += "\n**Especificaciones:**\n"
        for spec in tech_info['specifications'][:2]:
            response += f"• {spec}\n"
    
    return response

def generate_oil_response_with_context(tech_info, confidence):
    """Respuesta específica para aceite con contexto"""
    response = "**SISTEMA DE LUBRICACIÓN - Miguel Mecánico**\n\n"
    
    if tech_info['specifications']:
        response += "**Especificaciones del aceite:**\n"
        for spec in tech_info['specifications'][:3]:
            response += f"• {spec}\n"
        response += "\n"
    
    response += """**Cambio de aceite paso a paso:**
1. Motor tibio (no caliente) para mejor drenado
2. Drenar aceite usado completamente
3. Reemplazar filtro de aceite nuevo
4. Rellenar con aceite especificado
5. Verificar nivel después de 5 minutos

**Intervalos de mantenimiento:**
• Aceite sintético: 10,000-15,000 km
• Aceite convencional: 5,000-7,500 km
• Filtro: cada cambio de aceite
• Verificar nivel: semanalmente"""
    
    if tech_info['procedures']:
        response += "\n**Procedimientos adicionales:**\n"
        for proc in tech_info['procedures'][:2]:
            response += f"• {proc}\n"
    
    return response

def generate_electrical_response_with_context(tech_info, confidence):
    """Respuesta específica para sistema eléctrico con contexto"""
    response = "**SISTEMA ELÉCTRICO - Miguel Mecánico**\n\n"
    
    if tech_info['specifications']:
        response += "**Especificaciones eléctricas:**\n"
        for spec in tech_info['specifications'][:3]:
            response += f"• {spec}\n"
        response += "\n"
    
    response += """**Diagnóstico de batería y alternador:**
• Voltaje batería en reposo: 12.6V
• Voltaje con motor encendido: 13.8-14.4V
• Densidad del electrolito: 1.265 g/cm³
• Terminales limpios y apretados

**Pruebas básicas:**
1. Multímetro en terminales de batería
2. Prueba de carga (arranque del motor)
3. Verificar correa del alternador
4. Revisar conexiones a masa"""
    
    if tech_info['procedures']:
        response += "\n**Procedimientos específicos:**\n"
        for proc in tech_info['procedures'][:2]:
            response += f"• {proc}\n"
    
    return response

def generate_tires_response_with_context(tech_info, confidence):
    """Respuesta específica para neumáticos con contexto"""
    response = "**NEUMÁTICOS Y SUSPENSIÓN - Miguel Mecánico**\n\n"
    
    if tech_info['specifications']:
        response += "**Especificaciones:**\n"
        for spec in tech_info['specifications'][:3]:
            response += f"• {spec}\n"
        response += "\n"
    
    response += """**Presiones recomendadas (verificar etiqueta del vehículo):**
• Neumáticos delanteros: 32-35 PSI
• Neumáticos traseros: 30-33 PSI
• Rueda de repuesto: 60 PSI
• Verificación: neumático frío

**Inspección visual:**
• Profundidad de banda (mínimo 1.6mm)
• Desgaste uniforme en toda la banda
• Grietas en flancos
• Objetos clavados (clavos, tornillos)"""
    
    return response

def generate_transmission_response_with_context(tech_info, confidence):
    """Respuesta específica para transmisión con contexto"""
    response = "**TRANSMISIÓN - Miguel Mecánico**\n\n"
    
    if tech_info['procedures']:
        response += "**Procedimientos técnicos:**\n"
        for proc in tech_info['procedures'][:3]:
            response += f"• {proc}\n"
        response += "\n"
    
    response += """**Mantenimiento de transmisión:**
• Manual: Aceite cada 60,000 km
• Automática: Fluido cada 80,000 km
• Embrague: Inspección cada 40,000 km
• Verificar fugas regularmente

**Síntomas de problemas:**
• Dificultad para cambiar marchas
• Ruidos al cambiar
• Embrague que patina
• Tirones al acelerar"""
    
    if tech_info['specifications']:
        response += "\n**Especificaciones:**\n"
        for spec in tech_info['specifications'][:2]:
            response += f"• {spec}\n"
    
    return response

def generate_general_response_with_context(tech_info, query, confidence):
    """Respuesta general con contexto técnico"""
    response = "**CONSULTA TÉCNICA - Miguel Mecánico**\n\n"
    
    if tech_info['procedures']:
        response += "**Información técnica relevante:**\n"
        for proc in tech_info['procedures'][:3]:
            response += f"• {proc}\n"
        response += "\n"
    
    response += """**Diagnóstico general recomendado:**
1. Inspección visual completa
2. Verificación de códigos de error (OBD)
3. Pruebas específicas según síntomas
4. Consulta de manual técnico del vehículo

**Herramientas básicas necesarias:**
• Multímetro digital
• Scanner OBD2
• Juego de llaves métricas
• Manómetros de presión"""
    
    if tech_info['specifications']:
        response += "\n**Datos técnicos:**\n"
        for spec in tech_info['specifications'][:2]:
            response += f"• {spec}\n"
    
    if confidence < 0.8:
        response += "\n\n⚠️ **Recomendación:** Para un diagnóstico más preciso, es recomendable una inspección física del vehículo."
    
    return response

def improve_pronunciation(text):
    """Mejora la pronunciación de términos mecánicos para Miguel"""
    improvements = {
        'motor': 'mo-tor',
        'frenos': 'fre-nos',
        'transmisión': 'trans-mi-sión',
        'batería': 'ba-te-ría',
        'aceite': 'a-cei-te',
        'neumáticos': 'neu-má-ti-cos',
        'presión': 'pre-sión',
        'temperatura': 'tem-pe-ra-tu-ra',
        'voltaje': 'vol-ta-je',
        'diagnóstico': 'diag-nós-ti-co',
        'mantenimiento': 'man-te-ni-mien-to',
        'especificación': 'es-pe-ci-fi-ca-ción'
    }
    
    for word, pronunciation in improvements.items():
        # Agregar énfasis en términos técnicos importantes
        text = text.replace(word, f"<emphasis level=\"moderate\">{word}</emphasis>")
    
    return text

def add_natural_pauses(text):
    """Añadir pausas naturales para Miguel"""
    # Pausas más largas después de puntos para explicaciones técnicas
    text = re.sub(r'([.!?])', r'\1<break time="700ms"/>', text)
    # Pausas después de comas para listas técnicas
    text = re.sub(r'(,)', r'\1<break time="300ms"/>', text)
    # Pausas después de dos puntos para especificaciones
    text = re.sub(r'(:)', r'\1<break time="400ms"/>', text)
    # Pausas antes de advertencias importantes
    text = re.sub(r'(⚠️)', r'<break time="500ms"/>\1', text)
    
    return text

def create_ssml_text(text):
    """Crea texto SSML optimizado para voz de Miguel"""
    improved_text = improve_pronunciation(text)
    text_with_pauses = add_natural_pauses(improved_text)
    
    ssml = f"""
    <speak>
        <prosody rate="95%" pitch="-2%" volume="loud">
            <amazon:effect name="drc">
                <amazon:effect vocal-tract-length="+5%">
                    {text_with_pauses}
                </amazon:effect>
            </amazon:effect>
        </prosody>
    </speak>
    """
    
    return ssml.strip()

def create_generative_ssml(text):
    """Crea SSML optimizado específicamente para motor generativo de Miguel"""
    improved_text = improve_pronunciation(text)
    text_with_pauses = add_natural_pauses(improved_text)
    
    ssml = f"""
    <speak>
        <prosody rate="90%" pitch="-3%" volume="medium">
            <amazon:auto-breaths volume="x-soft" frequency="low">
                {text_with_pauses}
            </amazon:auto-breaths>
        </prosody>
    </speak>
    """
    
    return ssml.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/speak', methods=['POST'])
def speak_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            app.logger.error("AWS credentials not configured - usando modo navegador")
            return jsonify({
                'audioContent': None,
                'audioUrl': None,
                'useBrowserTTS': True,
                'text': text
            })
        
        polly = boto3.client('polly',
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION)
        
        # INTENTAR PRIMERO CON MOTOR GENERATIVO
        try:
            ssml_text = create_generative_ssml(text)
            app.logger.info("Sintetizando con motor GENERATIVO - Voz Miguel...")
            
            response = polly.synthesize_speech(
                Text=ssml_text,
                TextType='ssml',
                OutputFormat='mp3',
                VoiceId='Miguel',  # Voz masculina para Miguel el mecánico
                Engine='generative',
                LanguageCode='es-US',
                SampleRate='24000'
            )
            
            app.logger.info("Audio sintetizado correctamente con motor GENERATIVO")
            
            audio_data = response['AudioStream'].read()
            audio_content = base64.b64encode(audio_data).decode('utf-8')
            
            return jsonify({
                'audioContent': audio_content,
                'audioUrl': f"data:audio/mp3;base64,{audio_content}",
                'useBrowserTTS': False,
                'engine': 'generative'
            })
            
        except (BotoCoreError, ClientError) as generative_error:
            app.logger.warning(f"Motor generativo falló: {generative_error}")
            
            # FALLBACK a motor neuronal
            try:
                ssml_text = create_ssml_text(text)
                response = polly.synthesize_speech(
                    Text=ssml_text,
                    TextType='ssml',
                    OutputFormat='mp3',
                    VoiceId='Miguel',  # Voz masculina para Miguel
                    Engine='neural',
                    LanguageCode='es-US'
                )
                
                app.logger.info("Audio sintetizado correctamente con motor neuronal")
                
                audio_data = response['AudioStream'].read()
                audio_content = base64.b64encode(audio_data).decode('utf-8')
                
                return jsonify({
                    'audioContent': audio_content,
                    'audioUrl': f"data:audio/mp3;base64,{audio_content}",
                    'useBrowserTTS': False,
                    'engine': 'neural'
                })
                
            except (BotoCoreError, ClientError) as neural_error:
                app.logger.error(f"AWS Polly neural error: {neural_error}")
                
                # Fallback a voz estándar
                try:
                    response = polly.synthesize_speech(
                        Text=text,
                        OutputFormat='mp3',
                        VoiceId='Miguel'  # Voz masculina para Miguel
                    )
                    
                    audio_data = response['AudioStream'].read()
                    audio_content = base64.b64encode(audio_data).decode('utf-8')
                    
                    return jsonify({
                        'audioContent': audio_content,
                        'audioUrl': f"data:audio/mp3;base64,{audio_content}",
                        'useBrowserTTS': False,
                        'engine': 'standard'
                    })
                    
                except Exception as fallback_error:
                    app.logger.error(f"Fallback también falló: {fallback_error}")
                    return jsonify({
                        'audioContent': None,
                        'audioUrl': None,
                        'useBrowserTTS': True,
                        'text': text,
                        'error': str(neural_error)
                    })
            
    except Exception as e:
        app.logger.error(f"Exception in speak_text: {str(e)}")
        return jsonify({
            'audioContent': None,
            'audioUrl': None,
            'useBrowserTTS': True,
            'text': text,
            'error': str(e)
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Saludo inicial personalizado para Miguel
        message_lower = message.lower()
        if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes', 'saludos', 'iniciar', 'empezar']):
            response = """¡Hola! Soy Miguel, tu mecánico especializado en vehículos. Tengo acceso a una amplia base de conocimientos técnicos y manuales de mecánica automotriz.

**Puedo ayudarte con:**
• Diagnóstico de problemas del motor
• Mantenimiento preventivo y correctivo
• Sistema de frenos y suspensión
• Diagnóstico eléctrico y electrónico
• Transmisión manual y automática
• Sistema de refrigeración y lubricación
• Neumáticos y alineación

**Para un mejor diagnóstico, comparte:**
- Marca, modelo y año del vehículo
- Síntomas específicos que observas
- Códigos de error (si los tienes)

¿Qué problema tiene tu vehículo?"""
        
        # Consulta técnica - usar Pinecone
        else:
            app.logger.info(f"Miguel consultando base técnica: {message}")
            
            # Consultar Pinecone para información técnica
            pinecone_results = query_pinecone(message)
            
            if pinecone_results and len(pinecone_results.matches) > 0:
                # Generar respuesta usando el contexto técnico
                response = generate_response(message, pinecone_results)
                app.logger.info(f"Respuesta técnica generada con {len(pinecone_results.matches)} referencias")
            else:
                # Respuesta cuando no hay datos en Pinecone
                response = """Como mecánico, no encuentro información específica sobre eso en mi base de conocimientos actual.

**Te puedo ayudar de manera general:**
- Describe el problema con más detalle
- ¿Qué síntomas específicos observas?
- ¿Cuándo comenzó el problema?
- ¿Hay sonidos, olores o luces de advertencia?

**Mientras tanto, verifica:**
• Niveles de fluidos (aceite, refrigerante, frenos)
• Estado de la batería (12.6V en reposo)
• Presión de neumáticos según especificación
• Códigos de error en el tablero

¿Podrías darme más información específica del problema?"""
        
        return jsonify({
            'response': response,
            'end_call': False
        })
            
    except Exception as e:
        app.logger.error(f"Exception in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio de Miguel"""
    aws_configured = bool(AWS_ACCESS_KEY and AWS_SECRET_KEY)
    pinecone_configured = bool(PINECONE_API_KEY)
    
    # Verificar conexión a Pinecone
    pinecone_status = "Desconectado"
    pinecone_stats = None
    
    if pinecone_configured:
        try:
            init_pinecone()
            if index:
                pinecone_status = "Conectado"
                pinecone_stats = index.describe_index_stats()
        except:
            pinecone_status = "Error de conexión"
    
    return jsonify({
        'status': 'healthy',
        'agent_name': 'Miguel - Mecánico Especializado',
        'aws_configured': aws_configured,
        'pinecone_configured': pinecone_configured,
        'pinecone_status': pinecone_status,
        'pinecone_stats': pinecone_stats,
        'model_loaded': model is not None,
        'voice_service': 'Amazon Polly - Miguel (Mecánica Automotriz)',
        'embedding_model': 'all-MiniLM-L6-v2',
        'index_name': INDEX_NAME
    })

@app.route('/api/pinecone-status', methods=['GET'])
def pinecone_status():
    """Endpoint específico para verificar estado de Pinecone"""
    try:
        pinecone_ready = init_pinecone()
        stats = None
        
        if index:
            stats = index.describe_index_stats()
            
        return jsonify({
            'pinecone_ready': pinecone_ready,
            'index_exists': index is not None,
            'index_name': INDEX_NAME,
            'model_loaded': model is not None,
            'index_stats': stats,
            'agent': 'Miguel - Mecánico Automotriz'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-knowledge', methods=['POST'])
def add_knowledge():
    """Endpoint para agregar conocimientos a la base de datos de Miguel"""
    try:
        if not index or not model:
            return jsonify({'error': 'Pinecone no inicializado'}), 500
            
        data = request.json
        text = data.get('text', '')
        doc_id = data.get('id', '')
        metadata = data.get('metadata', {})
        
        if not text or not doc_id:
            return jsonify({'error': 'Texto e ID son requeridos'}), 400
        
        # Generar embedding
        embedding = get_embedding(text)
        if not embedding:
            return jsonify({'error': 'No se pudo generar embedding'}), 500
        
        # Agregar metadatos adicionales para mecánica
        metadata.update({
            'text': text,
            'agent': 'Miguel',
            'category': 'mecanica_automotriz',
            'added_date': str(datetime.now())
        })
        
        # Upsert a Pinecone
        index.upsert([(doc_id, embedding, metadata)])
        
        app.logger.info(f"Conocimiento agregado: {doc_id}")
        
        return jsonify({
            'success': True,
            'message': f'Conocimiento agregado exitosamente a la base de Miguel',
            'doc_id': doc_id
        })
        
    except Exception as e:
        app.logger.error(f"Error agregando conocimiento: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Inicializar Pinecone y modelo al iniciar la aplicación
    app.logger.info("Iniciando Miguel - Agente Mecánico Especializado")
    init_pinecone()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
