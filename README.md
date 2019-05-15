# SIMULATION TO REAL-WORLD TRANSFER
## ЗАДАЧА ПРОЕКТА 
Ключевая цель: улучшение, модификация существующих RL алгоритмов для свободного перемещения адаптера (робота) в виртуальной и реальной среде.

Среда

1) Реализовать нейронную сеть для сегментации среды с целью выявления разметок дорог и указателей 2) Результаты сегментации передавать как среду для адаптера (робота)

RL алгоритм

1) Модифицировать алгоритм для лучших результатов в реальной среде при обучении в виртуальной

СДЕЛАННАЯ РАБОТЫ
Первая часть работ направлена на изучение RL и обзора архитектур сегментации (FCN)

Изученные статьи
RL:

1) https://arxiv.org/pdf/1509.02971.pdf

2) https://arxiv.org/pdf/1312.5602.pdf

3) http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

4) https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

5) https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html

FCN:

1) https://arxiv.org/pdf/1411.4038.pdf

2) https://arxiv.org/pdf/1505.04597.pdf

3) https://arxiv.org/pdf/1512.03385.pdf

4) https://keras.io

### Подготовка среды разработки

1) Установка программного обеспечение для виртуальной среды http://www.ros.org

2) Знакомство с порталом облачных вычислений https://cloud.google.com

ВЫВОДЫ И СЦЕНАРИИ РЕАЛИЗАЦИИ
Первая проблема, которая есть: после обучения робота в виртуальной среде, он не показывает нужного результата в реальном режиме работы.

Проблемы:

1) Виртуальная среда vs реальная разные визуально

2) В реальном мире камера имеет фишай искажение

3) В реальной среде много шума

4) Недостаточный набор размеченных данных.

Первая идея - способ подачи и разметки среды. Обучить сеть до такой степени, чтобы она смогла сегментировать виртуальную и реальную среду с фишай искажением.

Варианты реализации:

1) Дообучить предобученную сеть данными из виртуальной и реальной среды

2) Сделать бинарный классификатор (виртуальная/ реальная среда)

3) Построить биекцию и перевести фишай изображения в обычные

4) Применить фильтр для размытия фона (как предобработка и избавление от шумов)

Стек технологий

1) Python – язык реализации, в редакторе Jupyter + виртуальное окружение Anaconda

2) Требуемые библиотеки: keras, pytorch, matplotlib, numpy

3) Обучение на гугл клауд, на базе tensorflow



