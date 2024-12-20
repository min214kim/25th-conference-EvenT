import 'package:flutter_dotenv/flutter_dotenv.dart';

// 공통 URL 생성 함수
String createUrl(String endpoint) {
  // .env 파일에서 API_URL 가져오기
  final String apiUrl = dotenv.get('API_URL');
  // 공통 URL + 경로를 합쳐 반환
  return '$apiUrl$endpoint';
}