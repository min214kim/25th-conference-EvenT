import 'package:flutter/material.dart';
import 'first_screen//signup_page.dart'; // SignUpPage를 가져오기
import 'first_screen//login.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '안녕하세요!\n내 옷장 탐색기 EvenT 입니다.',
              textAlign: TextAlign.center,
              style: TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                  color: Colors.black
              ),
            ),
            const SizedBox(height: 50), // 텍스트와 이미지 사이 간격
            Image.asset( // 이미지
              'assets/homepage_img.png',

            ),
            const SizedBox(height: 50), // 여백
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30.0), // 양옆 마진
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  elevation: 0, // 그림자 없애기
                  backgroundColor: const Color(0xFFB8A39F), // 버튼 색상
                  minimumSize : const Size(500,45), // 버튼 크기
                  shape: RoundedRectangleBorder(
                      borderRadius:  BorderRadius.circular(25) // 모서리 둥글게
                  ),
                ), // style
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => LoginPage(),
                    ),
                  );
                },
                child: const Text(
                  '로그인',
                  style: TextStyle(
                    fontSize: 17,
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 3), // 여백
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30.0), // 양옆 마진
              child: OutlinedButton(
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: Color(0xFFB8A39F), width: 1.5), // 테두리 설정
                  minimumSize: const Size(500, 45), // 버튼 크기
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25), // 모서리 둥글게
                  ),
                ),
                onPressed: () {
                  // 다른 페이지 이동
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => SignUpPage(),
                    ),
                  );
                },
                child: const Text(
                    '회원가입',
                    style: TextStyle(
                      fontSize: 17,
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                    )
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}