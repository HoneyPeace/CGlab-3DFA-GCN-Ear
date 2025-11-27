% ✅ 저장 경로 설정
dataDir = fullfile(pwd, 'EarData', 'Ear296_Korean', 'template-registered_data');
savePath = fullfile(pwd, 'EarData', 'Ear296_Korean', 'template_registered_data_mat', 'Ear296_Korean.mat');

% ✅ 저장 경로 폴더 없으면 생성
saveDir = fileparts(savePath);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% ✅ 파일 목록 추출 (*.ply 기준)
plyFiles = dir(fullfile(dataDir, '*.ply'));
fprintf("총 .ply 파일 개수: %d\n", numel(plyFiles));

% ✅ 셀 배열 초기화
shape_all = cell(1, numel(plyFiles));
landmark_position_select_all = cell(1, numel(plyFiles));
landmark_index_select_all = cell(1, numel(plyFiles));

validSampleCount = 0;  % 유효 샘플 카운트

% ✅ 파일마다 반복
for i = 1:numel(plyFiles)
    baseName = erase(plyFiles(i).name, '.ply');
    plyPath = fullfile(dataDir, [baseName, '.ply']);
    ascPath = fullfile(dataDir, [baseName, '.asc']);

    fprintf('[%03d] 처리 중: %s\n', i, baseName);

    % ❗ ASC 파일 없으면 건너뜀
    if ~isfile(ascPath)
        fprintf('⚠️ ASC 누락 → 건너뜀: %s\n', ascPath);
        continue;
    end

    try
        % ✅ 포인트 클라우드 로딩
        pc = pcread(plyPath);
        shape_all{validSampleCount + 1} = single(pc.Location);

        % ✅ 랜드마크 로딩
        landmark = single(readmatrix(ascPath));
        landmark_position_select_all{validSampleCount + 1} = landmark;

        % ✅ 인덱스 생성 (0부터 시작)
        landmark_index_select_all{validSampleCount + 1} = int32(0:size(landmark,1)-1);

        validSampleCount = validSampleCount + 1;
    catch ME
        fprintf('❌ 오류 발생: %s\n', baseName);
        disp(ME.message);
        continue;
    end
end

% ✅ 유효 샘플만 잘라서 저장
shape_all = shape_all(1:validSampleCount);
landmark_position_select_all = landmark_position_select_all(1:validSampleCount);
landmark_index_select_all = landmark_index_select_all(1:validSampleCount);

% ✅ 저장 (HDF5 기반 v7)
save(savePath, 'shape_all', 'landmark_position_select_all', 'landmark_index_select_all', '-v7');
fprintf("✅ 저장 완료: %s\n", savePath);
fprintf("총 유효 샘플 수: %d\n", validSampleCount);
