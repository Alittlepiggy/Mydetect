from flask_cors import CORS
from flask import Flask, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from image_processor import ImageProcessor
import base64
import logging
from logging.handlers import RotatingFileHandler
import socket
import time
import io
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=1)
logger.addHandler(handler)

app = Flask(__name__)

# CORS配置 - 允许所有来源访问
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "X-Total-Count"]
    }
})

# 应用配置
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max-limit
    JSON_AS_ASCII=False,  # 支持中文
    PROPAGATE_EXCEPTIONS=True,  # 错误传播
    TEMPLATES_AUTO_RELOAD=True,  # 模板自动重载
    PREFERRED_URL_SCHEME='http'  # 默认URL方案
)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=4)

# 创建处理器实例池
processor_pool = []
for _ in range(4):
    processor_pool.append(ImageProcessor())

def get_processor():
    """从处理器池中获取一个可用的处理器"""
    return processor_pool.pop() if processor_pool else ImageProcessor()

def return_processor(processor):
    """将处理器返回到池中"""
    if len(processor_pool) < 4:
        processor_pool.append(processor)

def get_host_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        logger.error(f"获取IP地址失败: {str(e)}")
        ip = '0.0.0.0'  # 使用0.0.0.0代替127.0.0.1
    finally:
        s.close()
    return ip

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def process_image_task(image_data, operation, params=None):
    """处理图像的异步任务"""
    processor = get_processor()
    try:
        # 将图像数据转换为OpenCV格式
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图像数据")
        
        # 设置处理器的当前图像
        processor.current_image = image
        processor.original_image = image.copy()
        
        # 如果有参数，更新处理器参数
        if params:
            if 'brightness' in params:
                processor.brightness = float(params['brightness'])
            if 'contrast' in params:
                processor.contrast = float(params['contrast'])
            if 'canny_low' in params:
                processor.canny_low = int(params['canny_low'])
            if 'canny_high' in params:
                processor.canny_high = int(params['canny_high'])
            if 'fft_radius' in params:
                processor.fft_radius = int(params['fft_radius'])
            if 'morph_size' in params:
                processor.morph_size = int(params['morph_size'])
            if 'detection_mode' in params:
                processor.detection_mode = params['detection_mode']
        
        # 根据操作类型处理图像
        result = None
        defects = None
        info = {}
        
        if operation == 'enhance':
            result = processor.enhance_image()
        elif operation == 'detect_edges':
            result = processor.detect_edges()
            # 如果启用了边缘连接，进行处理
            if params.get('edge_connect_enabled', False):
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                connected = processor.connect_edges(gray, 
                                                 params.get('min_threshold', 5),
                                                 params.get('max_threshold', 15))
                result = cv2.cvtColor(connected, cv2.COLOR_GRAY2BGR)
        elif operation == 'detect_defects':
            result, defects = processor.detect_defects_intelligent()
        elif operation == 'detect_ai':
            # 确保加载了分割模型（如果需要）
            if params.get('detection_mode') in ['segment', 'both'] and not processor.segment_model:
                try:
                    processor.load_segment_model()
                except Exception as e:
                    logger.warning(f"加载分割模型失败: {str(e)}")
            
            result, defects = processor.detect_defects_ai()
            # 添加检测模式和统计信息到返回结果
            info['detection_mode'] = processor.detection_mode
            if defects and 'stats' in defects:
                info['stats'] = defects['stats']
        elif operation == 'adjust':
            result = processor.adjust_brightness_contrast(
                processor.brightness, 
                processor.contrast
            )
        elif operation == 'clahe':
            result = processor.clahe_enhancement()
        elif operation == 'fft':
            result = processor.fft_filter()
        elif operation == 'morph':
            morph_type = params.get('morph_type', 'erode')
            kernel = np.ones((processor.morph_size, processor.morph_size), np.uint8)
            if morph_type == 'erode':
                result = cv2.erode(processor.current_image, kernel)
            elif morph_type == 'dilate':
                result = cv2.dilate(processor.current_image, kernel)
            elif morph_type == 'open':
                result = cv2.morphologyEx(processor.current_image, cv2.MORPH_OPEN, kernel)
            elif morph_type == 'close':
                result = cv2.morphologyEx(processor.current_image, cv2.MORPH_CLOSE, kernel)
            elif morph_type == 'gradient':
                result = cv2.morphologyEx(processor.current_image, cv2.MORPH_GRADIENT, kernel)
            
        # 计算图像信息
        if result is not None:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            info.update({
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'min': int(np.min(gray)),
                'max': int(np.max(gray)),
                'size': f"{result.shape[1]}x{result.shape[0]}"
            })
            
            # 如果有缺陷检测结果
            if defects:
                for defect_type in defects:
                    if defect_type != 'stats':  # 跳过统计信息
                        info[defect_type] = len(defects[defect_type])
            
            # 将结果转换为base64
            _, buffer = cv2.imencode('.jpg', result)
            return base64.b64encode(buffer).decode('utf-8'), info
            
    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        raise Exception(f"处理图像时出错: {str(e)}")
    finally:
        return_processor(processor)

@app.route('/')
def index():
    """主页路由"""
    try:
        response = make_response(render_template('index.html'))
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
        return response
    except Exception as e:
        logger.error(f"访问主页出错: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/process', methods=['POST', 'OPTIONS'])
def process():
    """处理图像路由"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
        return response
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择图片'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
            
        operation = request.form.get('operation', 'enhance')
        
        # 获取处理参数
        params = {
            'brightness': request.form.get('brightness', 0, type=float),
            'contrast': request.form.get('contrast', 1.0, type=float),
            'canny_low': request.form.get('canny_low', 50, type=int),
            'canny_high': request.form.get('canny_high', 150, type=int),
            'fft_radius': request.form.get('fft_radius', 30, type=int),
            'morph_size': request.form.get('morph_size', 3, type=int),
            'morph_type': request.form.get('morph_type', 'erode'),
            'edge_connect_enabled': request.form.get('edge_connect_enabled', 'false') == 'true',
            'min_threshold': request.form.get('min_threshold', 5, type=int),
            'max_threshold': request.form.get('max_threshold', 15, type=int),
            'detection_mode': request.form.get('detection_mode', 'segment')
        }
        
        # 直接从内存中读取图像数据
        image_data = file.read()
        
        # 异步处理图像
        future = executor.submit(process_image_task, image_data, operation, params)
        result_base64, info = future.result()
        
        response = jsonify({
            'result': result_base64,
            'info': info,
            'message': '处理成功'
        })
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Expose-Headers': 'Content-Type'
        })
        return response
        
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """健康检查接口"""
    try:
        host_ip = get_host_ip()
        return jsonify({
            'status': 'healthy',
            'server_ip': host_ip,
            'timestamp': time.time(),
            'debug_mode': app.debug,
            'workers': len(processor_pool),
            'max_workers': 4
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/config')
def get_config():
    """获取服务器配置信息"""
    try:
        return jsonify({
            'max_content_length': app.config['MAX_CONTENT_LENGTH'],
            'allowed_extensions': list({'png', 'jpg', 'jpeg', 'bmp'}),
            'server_ip': get_host_ip(),
            'cors_enabled': True,
            'max_workers': 4
        })
    except Exception as e:
        logger.error(f"获取配置信息失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """处理404错误"""
    return jsonify({'error': '请求的资源不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    host_ip = get_host_ip()
    logger.info(f"服务器启动于: http://{host_ip}:443")
    # 设置host为'0.0.0.0'允许外部访问，开启线程支持
    app.run(host='0.0.0.0', port=443, debug=False, threaded=True) 