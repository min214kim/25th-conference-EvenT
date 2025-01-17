import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart'; // dotenv
import 'home_page.dart';

void main() async{
  // dotenv 초기화: .env 파일 로드
  await dotenv.load(fileName: ".env");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.light().copyWith(
        scaffoldBackgroundColor: Colors.white,
      ),
      home: const HomePage(),
    );
  }
}

