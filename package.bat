@echo off
rd /s /q %~dp0\bin
md "bin\build" "bin\dist"
copy ".\radiomicstool.ico" ".\bin\dist\radiomicstool.ico"
pyinstaller main.spec --distpath=bin\dist --workpath=bin\build
pause